#include <TFile.h>
#include <TTree.h>
#include <TError.h>
#include <TH1F.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TString.h>
#include <TPad.h>
#include <TAxis.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>
#include <array>
#include <limits>

struct ClusterInfo {
    int peakChannel;
    int subPeakChannel;      // -1 if none
    int subSubPeakChannel;   // -1 if none
    int clusterSize;         // 1 / 2 / 3
    double clusterValue;     // sum of adcSub over cluster channels
    double clusterPosition;  // charge-weighted center using adcSub
    double peakLocalRMS;     // event-by-event local residual RMS at peak channel
};

struct EventData {
    std::vector<double> adcSub;            // adc - pedestal - event-by-event CMN
    std::vector<double> cmnASIC;           // 16 values, one per 64 channels
    std::vector<double> localResidualRMS;  // event-by-event local RMS over channels
    std::vector<ClusterInfo> clusters;
};

struct Dataset {
    std::string tag;
    std::string rootFileName;
    std::string txtFileName;
    std::string txtOutName;

    TFile* inFile = nullptr;
    TTree* inTree = nullptr;
    Int_t adc[1024];

    std::vector<double> pedestal;

    std::ofstream txtOut;
    TTree* outTree = nullptr;

    std::vector<EventData> events;

    // output tree branches
    Int_t eventID = -1;
    Int_t nClusters = 0;
    std::vector<int> peakChannel;
    std::vector<int> subPeakChannel;
    std::vector<int> subSubPeakChannel;
    std::vector<int> clusterSize;
    std::vector<double> clusterValue;
    std::vector<double> clusterPosition;
    std::vector<double> peakLocalRMS;
    std::vector<double> cmnASIC;
};

struct TreeInfo {
    TTree* tree;
    TString label;
    TString treeName;
};

struct ClusterPadObjects {
    TH1F* hSig = nullptr;
    TH1F* hRMS = nullptr;
    TGraph* gPeak = nullptr;
    TGraph* gSub = nullptr;
    TGraph* gSubSub = nullptr;
    TLegend* leg = nullptr;
};

// ============================================================================
// basic utilities
// ============================================================================

double MedianOfVector(std::vector<double> v)
{
    if (v.empty()) return 0.0;

    size_t n = v.size();
    size_t mid = n / 2;

    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double med = v[mid];

    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
        med = 0.5 * (med + v[mid - 1]);
    }
    return med;
}

double RobustSigmaMAD(const std::vector<double>& values)
{
    if (values.empty()) return 0.0;

    double med = MedianOfVector(values);

    std::vector<double> absDev(values.size(), 0.0);
    for (size_t i = 0; i < values.size(); ++i) {
        absDev[i] = std::fabs(values[i] - med);
    }

    double mad = MedianOfVector(absDev);
    double sigma = 1.4826 * mad;

    if (sigma < 1e-6) {
        double s2 = 0.0;
        for (double x : values) s2 += (x - med) * (x - med);
        sigma = std::sqrt(s2 / values.size());
    }
    if (sigma < 1e-6) sigma = 1e-6;

    return sigma;
}

std::vector<double> BuildLocalResidualRMS(const std::vector<double>& adcSub,
                                          int halfWindow = 32,
                                          int excludeCenter = 2)
{
    const int nChannels = (int)adcSub.size();
    std::vector<double> localRMS(nChannels, 0.0);

    double globalSigma = RobustSigmaMAD(adcSub);
    if (globalSigma < 1e-6) globalSigma = 1.0;

    for (int ch = 0; ch < nChannels; ++ch) {
        int lo = std::max(0, ch - halfWindow);
        int hi = std::min(nChannels - 1, ch + halfWindow);

        std::vector<double> localVals;
        localVals.reserve(hi - lo + 1);

        for (int i = lo; i <= hi; ++i) {
            if (std::abs(i - ch) <= excludeCenter) continue;
            localVals.push_back(adcSub[i]);
        }

        if ((int)localVals.size() < 8)
            localRMS[ch] = globalSigma;
        else
            localRMS[ch] = RobustSigmaMAD(localVals);

        if (localRMS[ch] < 1e-6) localRMS[ch] = globalSigma;
    }

    return localRMS;
}

bool LoadPedestalOnly(const char* txtFileName,
                      std::vector<double>& pedestal,
                      int nChannels = 1024)
{
    pedestal.assign(nChannels, 0.0);

    std::ifstream fin(txtFileName);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open txt file: " << txtFileName << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        int ch;
        double ped;
        if (!(iss >> ch >> ped)) continue;

        if (ch >= 0 && ch < nChannels) {
            pedestal[ch] = ped;
        }
    }

    fin.close();
    return true;
}

bool InitDataset(Dataset& ds, int nChannels = 1024)
{
    if (!LoadPedestalOnly(ds.txtFileName.c_str(), ds.pedestal, nChannels)) {
        return false;
    }

    ds.inFile = TFile::Open(ds.rootFileName.c_str(), "READ");
    if (!ds.inFile || ds.inFile->IsZombie()) {
        std::cerr << "Error: cannot open input ROOT file: " << ds.rootFileName << std::endl;
        return false;
    }

    ds.inTree = (TTree*)ds.inFile->Get("tree");
    if (!ds.inTree) {
        std::cerr << "Error: cannot find TTree 'tree' in " << ds.rootFileName << std::endl;
        return false;
    }

    ds.inTree->SetBranchAddress("adc", ds.adc);

    ds.txtOut.open(ds.txtOutName.c_str());
    if (!ds.txtOut.is_open()) {
        std::cerr << "Error: cannot open output txt file: " << ds.txtOutName << std::endl;
        return false;
    }

    return true;
}

void CloseDataset(Dataset& ds)
{
    if (ds.txtOut.is_open()) ds.txtOut.close();

    if (ds.inFile) {
        ds.inFile->Close();
        ds.inFile = nullptr;
    }
}

std::vector<double> ComputeResidualPed(const Int_t* adc,
                                       const std::vector<double>& pedestal,
                                       int nChannels = 1024)
{
    std::vector<double> residualPed(nChannels, 0.0);
    for (int i = 0; i < nChannels; ++i) {
        residualPed[i] = adc[i] - pedestal[i];
    }
    return residualPed;
}

std::vector<double> ComputeEventCMNFromMedian(const std::vector<double>& residualPed,
                                              int nASIC = 16,
                                              int channelsPerASIC = 64)
{
    std::vector<double> cmnASIC(nASIC, 0.0);

    for (int ia = 0; ia < nASIC; ++ia) {
        int lo = ia * channelsPerASIC;
        int hi = lo + channelsPerASIC;

        std::vector<double> blockVals;
        blockVals.reserve(channelsPerASIC);

        for (int ch = lo; ch < hi; ++ch) {
            blockVals.push_back(residualPed[ch]);
        }

        cmnASIC[ia] = MedianOfVector(blockVals);
    }

    return cmnASIC;
}

std::vector<double> BuildADCSubFromCMN(const std::vector<double>& residualPed,
                                       const std::vector<double>& cmnASIC,
                                       int nChannels = 1024,
                                       int channelsPerASIC = 64)
{
    std::vector<double> adcSub(nChannels, 0.0);

    for (int ch = 0; ch < nChannels; ++ch) {
        int asic = ch / channelsPerASIC;
        adcSub[ch] = residualPed[ch] - cmnASIC[asic];
    }
    return adcSub;
}

std::vector<double> ExpandCMNToChannels(const std::vector<double>& cmnASIC,
                                        int nChannels = 1024,
                                        int channelsPerASIC = 64)
{
    std::vector<double> expanded(nChannels, 0.0);
    for (int ch = 0; ch < nChannels; ++ch) {
        int asic = ch / channelsPerASIC;
        expanded[ch] = cmnASIC[asic];
    }
    return expanded;
}

double ComputeClusterPosition(const std::vector<int>& channels,
                              const std::vector<double>& adcSub)
{
    double weightedSum = 0.0;
    double sum = 0.0;

    for (int ch : channels) {
        weightedSum += adcSub[ch] * ch;
        sum += adcSub[ch];
    }

    if (sum == 0.0) return -999.0;
    return weightedSum / sum;
}

// ============================================================================
// cluster finding
// ============================================================================

std::vector<ClusterInfo> FindClusters(const std::vector<double>& adcSub,
                                      const std::vector<double>& localResidualRMS)
{
    const int nChannels = (int)adcSub.size();
    std::vector<ClusterInfo> clusters;
    if (nChannels < 5) return clusters;

    struct PeakCandidate {
        int ch;
        double value;
    };
    std::vector<PeakCandidate> candidates;

    for (int ch = 2; ch <= nChannels - 3; ++ch) {
        double val = adcSub[ch];
        double sigmaLocal = localResidualRMS[ch];

        if (val > 15.0 &&                         // <- requirement 1
            val > 5.0 * sigmaLocal &&
            val > 3.0 * adcSub[ch - 2] &&
            val > 3.0 * adcSub[ch + 2]) {
            candidates.push_back({ch, val});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const PeakCandidate& a, const PeakCandidate& b) {
                  return a.value > b.value;
              });

    std::vector<bool> blocked(nChannels, false);

    for (const auto& cand : candidates) {
        int ch = cand.ch;
        if (blocked[ch - 1] || blocked[ch] || blocked[ch + 1]) continue;

        int left  = ch - 1;
        int right = ch + 1;

        int subPeak = (adcSub[left] >= adcSub[right]) ? left : right;
        int smaller = (subPeak == left) ? right : left;

        ClusterInfo cl;
        cl.peakChannel = ch;
        cl.subPeakChannel = -1;
        cl.subSubPeakChannel = -1;
        cl.clusterSize = 1;
        cl.clusterValue = adcSub[ch];
        cl.peakLocalRMS = localResidualRMS[ch];

        std::vector<int> clusterChannels;
        clusterChannels.push_back(ch);

        if (adcSub[subPeak] >= 0.2 * adcSub[ch]) {
            cl.subPeakChannel = subPeak;
            cl.clusterSize = 2;
            cl.clusterValue += adcSub[subPeak];
            clusterChannels.push_back(subPeak);

            if (adcSub[smaller] > 3.0 * localResidualRMS[smaller]) {
                cl.subSubPeakChannel = smaller;
                cl.clusterSize = 3;
                cl.clusterValue += adcSub[smaller];
                clusterChannels.push_back(smaller);
            }
        }

        cl.clusterPosition = ComputeClusterPosition(clusterChannels, adcSub);
        clusters.push_back(cl);

        blocked[ch - 1] = true;
        blocked[ch]     = true;
        blocked[ch + 1] = true;
    }

    // at most two clusters; keep the two largest clusterValue if more than 2
    if ((int)clusters.size() > 2) {
        std::sort(clusters.begin(), clusters.end(),
                  [](const ClusterInfo& a, const ClusterInfo& b) {
                      return a.clusterValue > b.clusterValue;
                  });
        clusters.resize(2);
    }

    std::sort(clusters.begin(), clusters.end(),
              [](const ClusterInfo& a, const ClusterInfo& b) {
                  return a.peakChannel < b.peakChannel;
              });

    return clusters;
}

void PrecomputeAllEvents(Dataset& ds,
                         Long64_t nEvents,
                         int nChannels = 1024,
                         int nASIC = 16,
                         int channelsPerASIC = 64)
{
    ds.events.clear();
    ds.events.resize((size_t)nEvents);

    for (Long64_t ievt = 0; ievt < nEvents; ++ievt) {
        ds.inTree->GetEntry(ievt);

        std::vector<double> residualPed = ComputeResidualPed(ds.adc, ds.pedestal, nChannels);
        std::vector<double> cmnASIC = ComputeEventCMNFromMedian(residualPed, nASIC, channelsPerASIC);
        std::vector<double> adcSub = BuildADCSubFromCMN(residualPed, cmnASIC, nChannels, channelsPerASIC);
        std::vector<double> localResidualRMS = BuildLocalResidualRMS(adcSub, 32, 2);
        std::vector<ClusterInfo> clusters = FindClusters(adcSub, localResidualRMS);

        ds.events[(size_t)ievt].cmnASIC = cmnASIC;
        ds.events[(size_t)ievt].adcSub = adcSub;
        ds.events[(size_t)ievt].localResidualRMS = localResidualRMS;
        ds.events[(size_t)ievt].clusters = clusters;
    }
}

// ============================================================================
// output tree / txt
// ============================================================================

void SetupOutputTree(Dataset& ds, TFile* fout)
{
    fout->cd();

    TString treeName = Form("clusterTree_%s", ds.tag.c_str());
    TString treeTitle = Form("Cluster results for %s", ds.tag.c_str());

    ds.outTree = new TTree(treeName, treeTitle);

    ds.outTree->Branch("eventID", &ds.eventID);
    ds.outTree->Branch("nClusters", &ds.nClusters);
    ds.outTree->Branch("peakChannel", &ds.peakChannel);
    ds.outTree->Branch("subPeakChannel", &ds.subPeakChannel);
    ds.outTree->Branch("subSubPeakChannel", &ds.subSubPeakChannel);
    ds.outTree->Branch("clusterSize", &ds.clusterSize);
    ds.outTree->Branch("clusterValue", &ds.clusterValue);
    ds.outTree->Branch("clusterPosition", &ds.clusterPosition);
    ds.outTree->Branch("peakLocalRMS", &ds.peakLocalRMS);
    ds.outTree->Branch("cmnASIC", &ds.cmnASIC);
}

void FillOutputTree(Dataset& ds, int eventID, const EventData& ev)
{
    ds.eventID = eventID;
    ds.nClusters = (Int_t)ev.clusters.size();

    ds.peakChannel.clear();
    ds.subPeakChannel.clear();
    ds.subSubPeakChannel.clear();
    ds.clusterSize.clear();
    ds.clusterValue.clear();
    ds.clusterPosition.clear();
    ds.peakLocalRMS.clear();
    ds.cmnASIC = ev.cmnASIC;

    for (const auto& cl : ev.clusters) {
        ds.peakChannel.push_back(cl.peakChannel);
        ds.subPeakChannel.push_back(cl.subPeakChannel);
        ds.subSubPeakChannel.push_back(cl.subSubPeakChannel);
        ds.clusterSize.push_back(cl.clusterSize);
        ds.clusterValue.push_back(cl.clusterValue);
        ds.clusterPosition.push_back(cl.clusterPosition);
        ds.peakLocalRMS.push_back(cl.peakLocalRMS);
    }

    ds.outTree->Fill();
}

void WriteEventInfoToTxt(std::ofstream& txtout,
                         const std::string& tag,
                         int eventID,
                         const std::vector<double>& cmnASIC,
                         const std::vector<ClusterInfo>& clusters)
{
    txtout << tag << "  Event " << eventID << "\n";

    txtout << "  CMN(16 ASICs) = ";
    for (size_t i = 0; i < cmnASIC.size(); ++i) {
        txtout << cmnASIC[i];
        if (i + 1 != cmnASIC.size()) txtout << ", ";
    }
    txtout << "\n";

    txtout << "  nClusters = " << clusters.size() << "\n";

    for (size_t i = 0; i < clusters.size(); ++i) {
        const auto& cl = clusters[i];
        txtout << "  Cluster " << i
               << ": peakChannel = " << cl.peakChannel
               << ", subPeakChannel = " << cl.subPeakChannel
               << ", subSubPeakChannel = " << cl.subSubPeakChannel
               << ", clusterSize = " << cl.clusterSize
               << ", clusterValue = " << cl.clusterValue
               << ", clusterPosition = " << cl.clusterPosition
               << ", peakLocalRMS = " << cl.peakLocalRMS
               << "\n";
    }
    txtout << "\n";
}

void PrintEvent0Summary(const Dataset& ds)
{
    if (ds.events.empty()) return;

    const EventData& ev = ds.events[0];

    std::cout << "================ " << ds.tag << " Event 0 summary ================\n";
    std::cout << "CMN(16 ASICs): ";
    for (size_t i = 0; i < ev.cmnASIC.size(); ++i) {
        std::cout << ev.cmnASIC[i];
        if (i + 1 != ev.cmnASIC.size()) std::cout << ", ";
    }
    std::cout << "\n";

    std::cout << "Number of clusters = " << ev.clusters.size() << "\n";
    for (size_t i = 0; i < ev.clusters.size(); ++i) {
        const auto& cl = ev.clusters[i];
        std::cout << "Cluster " << i
                  << ": peakChannel = " << cl.peakChannel
                  << ", subPeakChannel = " << cl.subPeakChannel
                  << ", subSubPeakChannel = " << cl.subSubPeakChannel
                  << ", clusterSize = " << cl.clusterSize
                  << ", clusterValue = " << cl.clusterValue
                  << ", clusterPosition = " << cl.clusterPosition
                  << ", peakLocalRMS = " << cl.peakLocalRMS
                  << "\n";
    }
    std::cout << "=========================================================\n";
}

// ============================================================================
// drawing helpers
// ============================================================================

void PrintCanvasToPdf(TCanvas* c,
                      const char* outPdfName,
                      int pageIndex,
                      int totalPages)
{
    if (pageIndex == 0)
        c->Print(Form("%s(", outPdfName), "pdf");
    else if (pageIndex == totalPages - 1)
        c->Print(Form("%s)", outPdfName), "pdf");
    else
        c->Print(outPdfName, "pdf");
}

void SetHistStyle(TH1F* h, int color)
{
    h->SetLineColor(color);
    h->SetLineWidth(2);
    h->SetStats(0);
    h->GetXaxis()->SetTitleSize(0.05);
    h->GetYaxis()->SetTitleSize(0.05);
    h->GetXaxis()->SetLabelSize(0.045);
    h->GetYaxis()->SetLabelSize(0.045);
    h->GetYaxis()->SetTitleOffset(0.80);
}

void SetPadStyle(double bottomMargin = 0.08)
{
    gPad->SetLeftMargin(0.08);
    gPad->SetRightMargin(0.03);
    gPad->SetTopMargin(0.10);
    gPad->SetBottomMargin(bottomMargin);
}

void FillHistFromVector(TH1F* h, const std::vector<double>& vals)
{
    for (int i = 0; i < (int)vals.size(); ++i) {
        h->SetBinContent(i + 1, vals[i]);
    }
}

// ============================================================================
// per-event waveform / CMN / cluster-check pages
// ============================================================================

void DrawWaveformPage(const std::array<Dataset,3>& ds,
                      int eventID,
                      const char* outPdfName,
                      int pageIndex,
                      int totalPages)
{
    const int nChannels = 1024;

    TH1F* h1 = new TH1F(Form("hWave_%s_evt_%d", ds[0].tag.c_str(), eventID),
                        Form("%s Event %d;Channel;ADC - pedestal - CMN", ds[0].tag.c_str(), eventID),
                        nChannels, 0, nChannels);
    TH1F* h2 = new TH1F(Form("hWave_%s_evt_%d", ds[1].tag.c_str(), eventID),
                        Form("%s Event %d;Channel;ADC - pedestal - CMN", ds[1].tag.c_str(), eventID),
                        nChannels, 0, nChannels);
    TH1F* h3 = new TH1F(Form("hWave_%s_evt_%d", ds[2].tag.c_str(), eventID),
                        Form("%s Event %d;Channel;ADC - pedestal - CMN", ds[2].tag.c_str(), eventID),
                        nChannels, 0, nChannels);

    h1->SetDirectory(0);
    h2->SetDirectory(0);
    h3->SetDirectory(0);

    FillHistFromVector(h1, ds[0].events[eventID].adcSub);
    FillHistFromVector(h2, ds[1].events[eventID].adcSub);
    FillHistFromVector(h3, ds[2].events[eventID].adcSub);

    SetHistStyle(h1, kBlue + 1);
    SetHistStyle(h2, kRed + 1);
    SetHistStyle(h3, kBlack);

    double ymin =  1e30;
    double ymax = -1e30;
    TH1F* hs[3] = {h1, h2, h3};

    for (int ih = 0; ih < 3; ++ih) {
        for (int ib = 1; ib <= hs[ih]->GetNbinsX(); ++ib) {
            double val = hs[ih]->GetBinContent(ib);
            if (val < ymin) ymin = val;
            if (val > ymax) ymax = val;
        }
    }

    double margin = 0.08 * (ymax - ymin);
    if (margin <= 0) margin = 1.0;
    ymin -= margin;
    ymax += margin;

    // keep these lines but comment them out as requested
    // h1->SetMinimum(ymin); h1->SetMaximum(ymax);
    // h2->SetMinimum(ymin); h2->SetMaximum(ymax);
    // h3->SetMinimum(ymin); h3->SetMaximum(ymax);

    h1->SetMinimum(-20); h1->SetMaximum(100);
    h2->SetMinimum(-20); h2->SetMaximum(100);
    h3->SetMinimum(-20); h3->SetMaximum(100);

    TCanvas* c = new TCanvas(Form("cWave_evt_%d", eventID),
                             Form("Waveform Event %d", eventID),
                             1400, 1200);
    c->Divide(1, 3, 0.002, 0.002);

    c->cd(1); SetPadStyle(0.08); h1->Draw("PL");
    c->cd(2); SetPadStyle(0.08); h2->Draw("PL");
    c->cd(3); SetPadStyle(0.12); h3->Draw("PL");

    PrintCanvasToPdf(c, outPdfName, pageIndex, totalPages);

    delete c;
    delete h1;
    delete h2;
    delete h3;
}

void DrawCMNPage(const std::array<Dataset,3>& ds,
                 int eventID,
                 const char* outPdfName,
                 int pageIndex,
                 int totalPages)
{
    const int nChannels = 1024;

    std::vector<double> cmn1 = ExpandCMNToChannels(ds[0].events[eventID].cmnASIC);
    std::vector<double> cmn2 = ExpandCMNToChannels(ds[1].events[eventID].cmnASIC);
    std::vector<double> cmn3 = ExpandCMNToChannels(ds[2].events[eventID].cmnASIC);

    TH1F* h1 = new TH1F(Form("hCMN_%s_evt_%d", ds[0].tag.c_str(), eventID),
                        Form("%s Event %d CMN;Channel;CMN", ds[0].tag.c_str(), eventID),
                        nChannels, 0, nChannels);
    TH1F* h2 = new TH1F(Form("hCMN_%s_evt_%d", ds[1].tag.c_str(), eventID),
                        Form("%s Event %d CMN;Channel;CMN", ds[1].tag.c_str(), eventID),
                        nChannels, 0, nChannels);
    TH1F* h3 = new TH1F(Form("hCMN_%s_evt_%d", ds[2].tag.c_str(), eventID),
                        Form("%s Event %d CMN;Channel;CMN", ds[2].tag.c_str(), eventID),
                        nChannels, 0, nChannels);

    h1->SetDirectory(0);
    h2->SetDirectory(0);
    h3->SetDirectory(0);

    FillHistFromVector(h1, cmn1);
    FillHistFromVector(h2, cmn2);
    FillHistFromVector(h3, cmn3);

    SetHistStyle(h1, kBlue + 1);
    SetHistStyle(h2, kRed + 1);
    SetHistStyle(h3, kBlack);

    double ymin =  1e30;
    double ymax = -1e30;
    TH1F* hs[3] = {h1, h2, h3};

    for (int ih = 0; ih < 3; ++ih) {
        for (int ib = 1; ib <= hs[ih]->GetNbinsX(); ++ib) {
            double val = hs[ih]->GetBinContent(ib);
            if (val < ymin) ymin = val;
            if (val > ymax) ymax = val;
        }
    }

    double margin = 0.10 * (ymax - ymin);
    if (margin <= 0) margin = 1.0;
    ymin -= margin;
    ymax += margin;

    h1->SetMinimum(ymin); h1->SetMaximum(ymax);
    h2->SetMinimum(ymin); h2->SetMaximum(ymax);
    h3->SetMinimum(ymin); h3->SetMaximum(ymax);

    TCanvas* c = new TCanvas(Form("cCMN_evt_%d", eventID),
                             Form("CMN Event %d", eventID),
                             1400, 1200);
    c->Divide(1, 3, 0.002, 0.002);

    c->cd(1); SetPadStyle(0.08); h1->Draw("HIST");
    c->cd(2); SetPadStyle(0.08); h2->Draw("HIST");
    c->cd(3); SetPadStyle(0.12); h3->Draw("HIST");

    PrintCanvasToPdf(c, outPdfName, pageIndex, totalPages);

    delete c;
    delete h1;
    delete h2;
    delete h3;
}

ClusterPadObjects DrawSingleClusterPad(const Dataset& ds,
                                       int eventID,
                                       double yMin,
                                       double yMax,
                                       double bottomMargin = 0.08)
{
    const int nChannels = 1024;
    const EventData& ev = ds.events[eventID];

    gPad->SetLeftMargin(0.08);
    gPad->SetRightMargin(0.03);
    gPad->SetTopMargin(0.10);
    gPad->SetBottomMargin(bottomMargin);

    ClusterPadObjects obj;

    obj.hSig = new TH1F(Form("hSig_%s_evt_%d", ds.tag.c_str(), eventID),
                        Form("%s Event %d;Channel;ADC - pedestal - CMN",
                             ds.tag.c_str(), eventID),
                        nChannels, 0, nChannels);

    obj.hRMS = new TH1F(Form("hRMS_%s_evt_%d", ds.tag.c_str(), eventID),
                        "",
                        nChannels, 0, nChannels);

    obj.hSig->SetDirectory(0);
    obj.hRMS->SetDirectory(0);

    FillHistFromVector(obj.hSig, ev.adcSub);
    FillHistFromVector(obj.hRMS, ev.localResidualRMS);

    //obj.hSig->SetMinimum(yMin);
    //obj.hSig->SetMaximum(yMax);
    obj.hSig->SetMinimum(-20);
    obj.hSig->SetMaximum(100);
    obj.hSig->SetLineColor(kBlack);
    obj.hSig->SetLineWidth(2);
    obj.hSig->SetStats(0);

    obj.hRMS->SetLineColor(kMagenta + 2);
    obj.hRMS->SetLineWidth(2);
    obj.hRMS->SetLineStyle(1);
    obj.hRMS->SetStats(0);

    obj.hSig->GetXaxis()->SetTitleSize(0.05);
    obj.hSig->GetYaxis()->SetTitleSize(0.05);
    obj.hSig->GetXaxis()->SetLabelSize(0.045);
    obj.hSig->GetYaxis()->SetLabelSize(0.045);
    obj.hSig->GetYaxis()->SetTitleOffset(0.80);

    std::vector<double> xPeak, yPeak;
    std::vector<double> xSub, ySub;
    std::vector<double> xSubSub, ySubSub;

    for (const auto& cl : ev.clusters) {
        xPeak.push_back(cl.peakChannel + 0.5);
        yPeak.push_back(ev.adcSub[cl.peakChannel]);

        if (cl.subPeakChannel >= 0) {
            xSub.push_back(cl.subPeakChannel + 0.5);
            ySub.push_back(ev.adcSub[cl.subPeakChannel]);
        }

        if (cl.subSubPeakChannel >= 0) {
            xSubSub.push_back(cl.subSubPeakChannel + 0.5);
            ySubSub.push_back(ev.adcSub[cl.subSubPeakChannel]);
        }
    }

    if (!xPeak.empty()) {
        obj.gPeak = new TGraph((int)xPeak.size(), &xPeak[0], &yPeak[0]);
        obj.gPeak->SetMarkerStyle(20);
        obj.gPeak->SetMarkerSize(1.15);
        obj.gPeak->SetMarkerColor(kRed);
    }

    if (!xSub.empty()) {
        obj.gSub = new TGraph((int)xSub.size(), &xSub[0], &ySub[0]);
        obj.gSub->SetMarkerStyle(21);
        obj.gSub->SetMarkerSize(1.15);
        obj.gSub->SetMarkerColor(kBlue);
    }

    if (!xSubSub.empty()) {
        obj.gSubSub = new TGraph((int)xSubSub.size(), &xSubSub[0], &ySubSub[0]);
        obj.gSubSub->SetMarkerStyle(22);
        obj.gSubSub->SetMarkerSize(1.15);
        obj.gSubSub->SetMarkerColor(kGreen + 2);
    }

    obj.hSig->Draw("HIST");
    obj.hRMS->Draw("HIST SAME");
    if (obj.gPeak)   obj.gPeak->Draw("P SAME");
    if (obj.gSub)    obj.gSub->Draw("P SAME");
    if (obj.gSubSub) obj.gSubSub->Draw("P SAME");

    obj.leg = new TLegend(0.64, 0.62, 0.94, 0.90);
    obj.leg->SetBorderSize(0);
    obj.leg->SetFillStyle(0);
    obj.leg->AddEntry(obj.hSig, "ADC - pedestal - CMN", "l");
    obj.leg->AddEntry(obj.hRMS, "localResidualRMS", "l");
    if (obj.gPeak)   obj.leg->AddEntry(obj.gPeak,   "Peak", "p");
    if (obj.gSub)    obj.leg->AddEntry(obj.gSub,    "Sub-peak", "p");
    if (obj.gSubSub) obj.leg->AddEntry(obj.gSubSub, "Sub-sub-peak", "p");
    obj.leg->Draw();

    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.036);
    latex.DrawLatex(0.12, 0.91,
        Form("%s   Event %d   nClusters = %zu",
             ds.tag.c_str(), eventID, ev.clusters.size()));

    gPad->Update();
    return obj;
}

void DeleteClusterPadObjects(ClusterPadObjects& obj)
{
    delete obj.leg;     obj.leg = nullptr;
    delete obj.gPeak;   obj.gPeak = nullptr;
    delete obj.gSub;    obj.gSub = nullptr;
    delete obj.gSubSub; obj.gSubSub = nullptr;
    delete obj.hSig;    obj.hSig = nullptr;
    delete obj.hRMS;    obj.hRMS = nullptr;
}

void DrawClusterPage(const std::array<Dataset,3>& ds,
                     int eventID,
                     const char* outPdfName,
                     int pageIndex,
                     int totalPages)
{
    double yMin =  1e30;
    double yMax = -1e30;

    for (int k = 0; k < 3; ++k) {
        const auto& sig = ds[k].events[eventID].adcSub;
        const auto& rms = ds[k].events[eventID].localResidualRMS;

        for (int ch = 0; ch < 1024; ++ch) {
            yMin = std::min(yMin, sig[ch]);
            yMax = std::max(yMax, sig[ch]);
            yMax = std::max(yMax, rms[ch]);
        }
    }

    double margin = 0.08 * (yMax - yMin);
    if (margin <= 0) margin = 3.0;
    yMin -= margin;
    yMax += margin;

    TCanvas* c = new TCanvas(Form("cCluster_evt_%d", eventID),
                             Form("Cluster Event %d", eventID),
                             1400, 1200);
    c->Divide(1, 3, 0.002, 0.002);

    c->cd(1);
    ClusterPadObjects obj1 = DrawSingleClusterPad(ds[0], eventID, yMin, yMax, 0.08);

    c->cd(2);
    ClusterPadObjects obj2 = DrawSingleClusterPad(ds[1], eventID, yMin, yMax, 0.08);

    c->cd(3);
    ClusterPadObjects obj3 = DrawSingleClusterPad(ds[2], eventID, yMin, yMax, 0.12);

    PrintCanvasToPdf(c, outPdfName, pageIndex, totalPages);

    DeleteClusterPadObjects(obj1);
    DeleteClusterPadObjects(obj2);
    DeleteClusterPadObjects(obj3);
    delete c;
}

// ============================================================================
// processing one 3-layer group
// ============================================================================

bool ProcessGroup(std::array<Dataset,3>& ds,
                  const std::string& outRootName,
                  const std::string& outPdfWaveName,
                  const std::string& outPdfCMNName,
                  const std::string& outPdfClusterCheckName)
{
    const int nChannels = 1024;
    const int nASIC = 16;
    const int channelsPerASIC = 64;

    for (int k = 0; k < 3; ++k) {
        if (!InitDataset(ds[k], nChannels)) {
            for (int j = 0; j <= k; ++j) CloseDataset(ds[j]);
            return false;
        }
    }

    Long64_t nEvents = ds[0].inTree->GetEntries();
    for (int k = 1; k < 3; ++k) {
        nEvents = std::min(nEvents, ds[k].inTree->GetEntries());
    }

    std::cout << "============================================================\n";
    std::cout << "Processing group: "
              << ds[0].tag << ", " << ds[1].tag << ", " << ds[2].tag << "\n";
    std::cout << ds[0].tag << " entries = " << ds[0].inTree->GetEntries() << std::endl;
    std::cout << ds[1].tag << " entries = " << ds[1].inTree->GetEntries() << std::endl;
    std::cout << ds[2].tag << " entries = " << ds[2].inTree->GetEntries() << std::endl;
    std::cout << "Will process min entries = " << nEvents << std::endl;

    if (nEvents <= 0) {
        std::cerr << "Error: no events to process.\n";
        for (int k = 0; k < 3; ++k) CloseDataset(ds[k]);
        return false;
    }

    for (int k = 0; k < 3; ++k) {
        std::cout << "Precomputing CMN and clusters for " << ds[k].tag << " ...\n";
        PrecomputeAllEvents(ds[k], nEvents, nChannels, nASIC, channelsPerASIC);
    }

    for (int k = 0; k < 3; ++k) {
        PrintEvent0Summary(ds[k]);
    }

    TFile* fout = TFile::Open(outRootName.c_str(), "RECREATE");
    if (!fout || fout->IsZombie()) {
        std::cerr << "Error: cannot create output ROOT file " << outRootName << std::endl;
        for (int k = 0; k < 3; ++k) CloseDataset(ds[k]);
        return false;
    }

    for (int k = 0; k < 3; ++k) {
        SetupOutputTree(ds[k], fout);
    }

    int totalPages = (int)nEvents;

    for (Long64_t ievt = 0; ievt < nEvents; ++ievt) {
        for (int k = 0; k < 3; ++k) {
            FillOutputTree(ds[k], (int)ievt, ds[k].events[(size_t)ievt]);
            WriteEventInfoToTxt(ds[k].txtOut,
                                ds[k].tag,
                                (int)ievt,
                                ds[k].events[(size_t)ievt].cmnASIC,
                                ds[k].events[(size_t)ievt].clusters);
        }

        DrawWaveformPage(ds, (int)ievt, outPdfWaveName.c_str(), (int)ievt, totalPages);
        DrawCMNPage(ds,      (int)ievt, outPdfCMNName.c_str(),  (int)ievt, totalPages);
        DrawClusterPage(ds,  (int)ievt, outPdfClusterCheckName.c_str(), (int)ievt, totalPages);
    }

    fout->cd();
    for (int k = 0; k < 3; ++k) {
        ds[k].outTree->Write();
    }
    fout->Close();
    delete fout;

    for (int k = 0; k < 3; ++k) {
        CloseDataset(ds[k]);
    }

    std::cout << "Output ROOT file saved to: " << outRootName << std::endl;
    std::cout << "Waveform PDF saved to: " << outPdfWaveName << std::endl;
    std::cout << "CMN PDF saved to: " << outPdfCMNName << std::endl;
    std::cout << "Cluster-check PDF saved to: " << outPdfClusterCheckName << std::endl;
    std::cout << "============================================================\n";

    return true;
}

// ============================================================================
// summary plots: merged distributions
// ============================================================================

bool CheckTrees(const std::vector<TreeInfo>& trees, const char* branchName)
{
    bool ok = true;
    for (const auto& info : trees) {
        if (!info.tree) {
            std::cerr << "Error: cannot find TTree " << info.treeName << std::endl;
            ok = false;
            continue;
        }
        if (!info.tree->GetBranch(branchName)) {
            std::cerr << "Error: branch \"" << branchName
                      << "\" not found in tree " << info.treeName << std::endl;
            ok = false;
        }
    }
    return ok;
}

// ----- vector<double> branch -----

bool GetGlobalMinMaxFromVectorDoubleBranch(const std::vector<TreeInfo>& trees,
                                           const char* branchName,
                                           double& xmin,
                                           double& xmax)
{
    xmin =  std::numeric_limits<double>::max();
    xmax = -std::numeric_limits<double>::max();
    bool found = false;

    for (const auto& info : trees) {
        std::vector<double>* vec = nullptr;
        info.tree->SetBranchStatus("*", 0);
        info.tree->SetBranchStatus(branchName, 1);
        info.tree->SetBranchAddress(branchName, &vec);

        Long64_t nEntries = info.tree->GetEntries();
        for (Long64_t i = 0; i < nEntries; ++i) {
            info.tree->GetEntry(i);
            if (!vec) continue;

            for (size_t j = 0; j < vec->size(); ++j) {
                double val = vec->at(j);
                xmin = std::min(xmin, val);
                xmax = std::max(xmax, val);
                found = true;
            }
        }

        info.tree->ResetBranchAddresses();
        info.tree->SetBranchStatus("*", 1);
    }

    if (!found) return false;

    if (xmin == xmax) {
        xmin -= 0.5;
        xmax += 0.5;
    } else {
        double pad = 0.05 * (xmax - xmin);
        xmin -= pad;
        xmax += pad;
    }

    return true;
}

TH1D* FillMergedHistogramVectorDouble(const std::vector<TreeInfo>& trees,
                                      const char* branchName,
                                      const TString& histName,
                                      int nbins,
                                      double xmin,
                                      double xmax,
                                      Color_t color)
{
    TH1D* h = new TH1D(histName, "", nbins, xmin, xmax);
    h->SetDirectory(0);
    h->SetLineColor(color);
    h->SetLineWidth(2);
    h->SetStats(0);

    for (const auto& info : trees) {
        std::vector<double>* vec = nullptr;
        info.tree->SetBranchStatus("*", 0);
        info.tree->SetBranchStatus(branchName, 1);
        info.tree->SetBranchAddress(branchName, &vec);

        Long64_t nEntries = info.tree->GetEntries();
        for (Long64_t i = 0; i < nEntries; ++i) {
            info.tree->GetEntry(i);
            if (!vec) continue;

            for (size_t j = 0; j < vec->size(); ++j) {
                h->Fill(vec->at(j));
            }
        }

        info.tree->ResetBranchAddresses();
        info.tree->SetBranchStatus("*", 1);
    }

    return h;
}

// ----- vector<int> branch -----

bool GetGlobalMinMaxFromVectorIntBranch(const std::vector<TreeInfo>& trees,
                                        const char* branchName,
                                        int& xmin,
                                        int& xmax)
{
    xmin =  std::numeric_limits<int>::max();
    xmax = -std::numeric_limits<int>::max();
    bool found = false;

    for (const auto& info : trees) {
        std::vector<int>* vec = nullptr;
        info.tree->SetBranchStatus("*", 0);
        info.tree->SetBranchStatus(branchName, 1);
        info.tree->SetBranchAddress(branchName, &vec);

        Long64_t nEntries = info.tree->GetEntries();
        for (Long64_t i = 0; i < nEntries; ++i) {
            info.tree->GetEntry(i);
            if (!vec) continue;

            for (size_t j = 0; j < vec->size(); ++j) {
                int val = vec->at(j);
                xmin = std::min(xmin, val);
                xmax = std::max(xmax, val);
                found = true;
            }
        }

        info.tree->ResetBranchAddresses();
        info.tree->SetBranchStatus("*", 1);
    }

    return found;
}

TH1D* FillMergedHistogramVectorInt(const std::vector<TreeInfo>& trees,
                                   const char* branchName,
                                   const TString& histName,
                                   int nbins,
                                   double xmin,
                                   double xmax,
                                   Color_t color)
{
    TH1D* h = new TH1D(histName, "", nbins, xmin, xmax);
    h->SetDirectory(0);
    h->SetLineColor(color);
    h->SetLineWidth(2);
    h->SetStats(0);

    for (const auto& info : trees) {
        std::vector<int>* vec = nullptr;
        info.tree->SetBranchStatus("*", 0);
        info.tree->SetBranchStatus(branchName, 1);
        info.tree->SetBranchAddress(branchName, &vec);

        Long64_t nEntries = info.tree->GetEntries();
        for (Long64_t i = 0; i < nEntries; ++i) {
            info.tree->GetEntry(i);
            if (!vec) continue;

            for (size_t j = 0; j < vec->size(); ++j) {
                h->Fill(vec->at(j));
            }
        }

        info.tree->ResetBranchAddresses();
        info.tree->SetBranchStatus("*", 1);
    }

    return h;
}

// ----- scalar int branch -----

bool GetGlobalMinMaxFromScalarIntBranch(const std::vector<TreeInfo>& trees,
                                        const char* branchName,
                                        int& xmin,
                                        int& xmax)
{
    xmin =  std::numeric_limits<int>::max();
    xmax = -std::numeric_limits<int>::max();
    bool found = false;

    for (const auto& info : trees) {
        Int_t val = 0;
        info.tree->SetBranchStatus("*", 0);
        info.tree->SetBranchStatus(branchName, 1);
        info.tree->SetBranchAddress(branchName, &val);

        Long64_t nEntries = info.tree->GetEntries();
        for (Long64_t i = 0; i < nEntries; ++i) {
            info.tree->GetEntry(i);
            xmin = std::min(xmin, (int)val);
            xmax = std::max(xmax, (int)val);
            found = true;
        }

        info.tree->ResetBranchAddresses();
        info.tree->SetBranchStatus("*", 1);
    }

    return found;
}

TH1D* FillMergedHistogramScalarInt(const std::vector<TreeInfo>& trees,
                                   const char* branchName,
                                   const TString& histName,
                                   int nbins,
                                   double xmin,
                                   double xmax,
                                   Color_t color)
{
    TH1D* h = new TH1D(histName, "", nbins, xmin, xmax);
    h->SetDirectory(0);
    h->SetLineColor(color);
    h->SetLineWidth(2);
    h->SetStats(0);

    for (const auto& info : trees) {
        Int_t val = 0;
        info.tree->SetBranchStatus("*", 0);
        info.tree->SetBranchStatus(branchName, 1);
        info.tree->SetBranchAddress(branchName, &val);

        Long64_t nEntries = info.tree->GetEntries();
        for (Long64_t i = 0; i < nEntries; ++i) {
            info.tree->GetEntry(i);
            h->Fill(val);
        }

        info.tree->ResetBranchAddresses();
        info.tree->SetBranchStatus("*", 1);
    }

    return h;
}

void DrawStatsLegend(TH1D* h)
{
    TLegend* leg = new TLegend(0.62, 0.72, 0.88, 0.88);
    leg->SetBorderSize(1);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.032);

    leg->AddEntry((TObject*)0, Form("Entries = %.0f", h->GetEntries()), "");
    leg->AddEntry((TObject*)0, Form("Mean = %.3f", h->GetMean()), "");
    leg->AddEntry((TObject*)0, Form("Std Dev = %.3f", h->GetStdDev()), "");
    leg->Draw();
}

void DrawSingleHist(TCanvas* c,
                    TH1D* h,
                    const TString& title,
                    const char* xTitle)
{
    c->cd();
    h->SetTitle(title);
    h->GetXaxis()->SetTitle(xTitle);
    h->GetYaxis()->SetTitle("Entries");
    h->Draw("hist");
    DrawStatsLegend(h);
    c->SetGrid();
    c->Update();
}

void DrawMergedVectorDoubleSummaryPdf(const std::vector<TreeInfo>& group1,
                                      const std::vector<TreeInfo>& group2,
                                      const std::vector<TreeInfo>& allTrees,
                                      const char* branchName,
                                      const TString& quantityLabel,
                                      const char* xTitle,
                                      const char* outPdf,
                                      int nbins)
{
    if (!CheckTrees(group1, branchName)) return;
    if (!CheckTrees(group2, branchName)) return;

    double xmin = 0.0, xmax = 0.0;
    if (!GetGlobalMinMaxFromVectorDoubleBranch(allTrees, branchName, xmin, xmax)) {
        std::cerr << "Error: failed to determine min/max for " << branchName << std::endl;
        return;
    }

    TH1D* h1 = FillMergedHistogramVectorDouble(group1, branchName, "h_vd_g1", nbins, xmin, xmax, kBlue+1);
    TH1D* h2 = FillMergedHistogramVectorDouble(group2, branchName, "h_vd_g2", nbins, xmin, xmax, kRed+1);
    TH1D* h3 = FillMergedHistogramVectorDouble(allTrees, branchName, "h_vd_all", nbins, xmin, xmax, kBlack);

    TCanvas* c1 = new TCanvas("c_vd_1", "c_vd_1", 900, 700);
    DrawSingleHist(c1, h1, quantityLabel + ": TB02 + TB05 + TB11", xTitle);

    TCanvas* c2 = new TCanvas("c_vd_2", "c_vd_2", 900, 700);
    DrawSingleHist(c2, h2, quantityLabel + ": TB07 + TB06 + TB10", xTitle);

    TCanvas* c3 = new TCanvas("c_vd_3", "c_vd_3", 900, 700);
    DrawSingleHist(c3, h3, quantityLabel + ": all six layers", xTitle);

    c1->Print((TString(outPdf) + "(").Data());
    c2->Print(outPdf);
    c3->Print((TString(outPdf) + ")").Data());

    delete c1; delete c2; delete c3;
    delete h1; delete h2; delete h3;
}

void DrawMergedVectorIntSummaryPdf(const std::vector<TreeInfo>& group1,
                                   const std::vector<TreeInfo>& group2,
                                   const std::vector<TreeInfo>& allTrees,
                                   const char* branchName,
                                   const TString& quantityLabel,
                                   const char* xTitle,
                                   const char* outPdf)
{
    if (!CheckTrees(group1, branchName)) return;
    if (!CheckTrees(group2, branchName)) return;

    int xmin = 0, xmax = 0;
    if (!GetGlobalMinMaxFromVectorIntBranch(allTrees, branchName, xmin, xmax)) {
        std::cerr << "Error: failed to determine min/max for " << branchName << std::endl;
        return;
    }

    double xlo = xmin - 0.5;
    double xhi = xmax + 0.5;
    int nbins = std::max(1, xmax - xmin + 1);

    TH1D* h1 = FillMergedHistogramVectorInt(group1, branchName, "h_vi_g1", nbins, xlo, xhi, kBlue+1);
    TH1D* h2 = FillMergedHistogramVectorInt(group2, branchName, "h_vi_g2", nbins, xlo, xhi, kRed+1);
    TH1D* h3 = FillMergedHistogramVectorInt(allTrees, branchName, "h_vi_all", nbins, xlo, xhi, kBlack);

    TCanvas* c1 = new TCanvas("c_vi_1", "c_vi_1", 900, 700);
    DrawSingleHist(c1, h1, quantityLabel + ": TB02 + TB05 + TB11", xTitle);

    TCanvas* c2 = new TCanvas("c_vi_2", "c_vi_2", 900, 700);
    DrawSingleHist(c2, h2, quantityLabel + ": TB07 + TB06 + TB10", xTitle);

    TCanvas* c3 = new TCanvas("c_vi_3", "c_vi_3", 900, 700);
    DrawSingleHist(c3, h3, quantityLabel + ": all six layers", xTitle);

    c1->Print((TString(outPdf) + "(").Data());
    c2->Print(outPdf);
    c3->Print((TString(outPdf) + ")").Data());

    delete c1; delete c2; delete c3;
    delete h1; delete h2; delete h3;
}

void DrawMergedScalarIntSummaryPdf(const std::vector<TreeInfo>& group1,
                                   const std::vector<TreeInfo>& group2,
                                   const std::vector<TreeInfo>& allTrees,
                                   const char* branchName,
                                   const TString& quantityLabel,
                                   const char* xTitle,
                                   const char* outPdf)
{
    if (!CheckTrees(group1, branchName)) return;
    if (!CheckTrees(group2, branchName)) return;

    int xmin = 0, xmax = 0;
    if (!GetGlobalMinMaxFromScalarIntBranch(allTrees, branchName, xmin, xmax)) {
        std::cerr << "Error: failed to determine min/max for " << branchName << std::endl;
        return;
    }

    double xlo = xmin - 0.5;
    double xhi = xmax + 0.5;
    int nbins = std::max(1, xmax - xmin + 1);

    TH1D* h1 = FillMergedHistogramScalarInt(group1, branchName, "h_si_g1", nbins, xlo, xhi, kBlue+1);
    TH1D* h2 = FillMergedHistogramScalarInt(group2, branchName, "h_si_g2", nbins, xlo, xhi, kRed+1);
    TH1D* h3 = FillMergedHistogramScalarInt(allTrees, branchName, "h_si_all", nbins, xlo, xhi, kBlack);

    TCanvas* c1 = new TCanvas("c_si_1", "c_si_1", 900, 700);
    DrawSingleHist(c1, h1, quantityLabel + ": TB02 + TB05 + TB11", xTitle);

    TCanvas* c2 = new TCanvas("c_si_2", "c_si_2", 900, 700);
    DrawSingleHist(c2, h2, quantityLabel + ": TB07 + TB06 + TB10", xTitle);

    TCanvas* c3 = new TCanvas("c_si_3", "c_si_3", 900, 700);
    DrawSingleHist(c3, h3, quantityLabel + ": all six layers", xTitle);

    c1->Print((TString(outPdf) + "(").Data());
    c2->Print(outPdf);
    c3->Print((TString(outPdf) + ")").Data());

    delete c1; delete c2; delete c3;
    delete h1; delete h2; delete h3;
}

// ============================================================================
// efficiency plot
// ============================================================================

double ComputeClusterEfficiency(TTree* tree)
{
    if (!tree) return 0.0;

    Int_t nClusters = 0;
    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("nClusters", 1);
    tree->SetBranchAddress("nClusters", &nClusters);

    Long64_t nEntries = tree->GetEntries();
    Long64_t good = 0;

    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        if (nClusters >= 1) ++good;
    }

    tree->ResetBranchAddresses();
    tree->SetBranchStatus("*", 1);

    if (nEntries <= 0) return 0.0;
    return (double)good / (double)nEntries;
}

void DrawEfficiencyPage(const std::vector<TreeInfo>& trees,
                        const TString& title,
                        const char* outPdf,
                        int pageIndex,
                        int totalPages)
{
    TH1D* h = new TH1D(Form("h_eff_%d", pageIndex), "", (int)trees.size(), 0.5, trees.size() + 0.5);
    h->SetDirectory(0);
    h->SetStats(0);
    h->SetLineWidth(2);
    h->SetMinimum(0.0);
    h->SetMaximum(1.05);
    h->SetTitle(title);
    h->GetYaxis()->SetTitle("Cluster reconstruction efficiency");
    h->GetXaxis()->SetTitle("Detector layer");
    h->GetYaxis()->SetTitleSize(0.05);
    h->GetXaxis()->SetTitleSize(0.05);
    h->GetYaxis()->SetLabelSize(0.045);
    h->GetXaxis()->SetLabelSize(0.045);
    h->GetYaxis()->SetTitleOffset(0.95);

    for (int i = 0; i < (int)trees.size(); ++i) {
        double eff = ComputeClusterEfficiency(trees[i].tree);
        h->SetBinContent(i + 1, eff);
        h->GetXaxis()->SetBinLabel(i + 1, trees[i].label);
    }

    TCanvas* c = new TCanvas(Form("c_eff_%d", pageIndex), Form("c_eff_%d", pageIndex), 900, 700);
    c->SetGridy();
    h->Draw("HIST");

    TLatex latex;
    latex.SetTextSize(0.035);
    latex.SetTextAlign(22);

    for (int i = 1; i <= h->GetNbinsX(); ++i) {
        double y = h->GetBinContent(i);
        latex.DrawLatex(i, y + 0.03, Form("%.3f", y));
    }

    PrintCanvasToPdf(c, outPdf, pageIndex, totalPages);

    delete c;
    delete h;
}

// ============================================================================
// summary controller
// ============================================================================

void DrawSummaryPlots(const char* rootY, const char* rootX)
{
    TFile* fY = TFile::Open(rootY, "READ");
    TFile* fX = TFile::Open(rootX, "READ");

    if (!fY || fY->IsZombie()) {
        std::cerr << "Error: cannot open " << rootY << std::endl;
        if (fX) { fX->Close(); delete fX; }
        return;
    }
    if (!fX || fX->IsZombie()) {
        std::cerr << "Error: cannot open " << rootX << std::endl;
        fY->Close(); delete fY;
        return;
    }

    std::vector<TreeInfo> groupY = {
        {(TTree*)fY->Get("clusterTree_TB02"), "TB02", "clusterTree_TB02"},
        {(TTree*)fY->Get("clusterTree_TB05"), "TB05", "clusterTree_TB05"},
        {(TTree*)fY->Get("clusterTree_TB11"), "TB11", "clusterTree_TB11"}
    };

    std::vector<TreeInfo> groupX = {
        {(TTree*)fX->Get("clusterTree_TB07"), "TB07", "clusterTree_TB07"},
        {(TTree*)fX->Get("clusterTree_TB06"), "TB06", "clusterTree_TB06"},
        {(TTree*)fX->Get("clusterTree_TB10"), "TB10", "clusterTree_TB10"}
    };

    std::vector<TreeInfo> allTrees = groupY;
    allTrees.insert(allTrees.end(), groupX.begin(), groupX.end());

    DrawMergedVectorDoubleSummaryPdf(groupY, groupX, allTrees,
                                     "clusterValue",
                                     "clusterValue",
                                     "clusterValue",
                                     "clusterValue_merged.pdf",
                                     200);

    DrawMergedVectorIntSummaryPdf(groupY, groupX, allTrees,
                                  "clusterSize",
                                  "clusterSize",
                                  "clusterSize",
                                  "clusterSize_merged.pdf");

    DrawMergedScalarIntSummaryPdf(groupY, groupX, allTrees,
                                  "nClusters",
                                  "cluster number",
                                  "nClusters",
                                  "clusterNumber_merged.pdf");

    const char* effPdf = "clusterEfficiency.pdf";
    DrawEfficiencyPage(groupX, "Cluster reconstruction efficiency: X direction", effPdf, 0, 2);
    DrawEfficiencyPage(groupY, "Cluster reconstruction efficiency: Y direction", effPdf, 1, 2);

    std::cout << "Saved summary PDFs:\n"
              << "  clusterValue_merged.pdf\n"
              << "  clusterSize_merged.pdf\n"
              << "  clusterNumber_merged.pdf\n"
              << "  clusterEfficiency.pdf\n";

    fY->Close();
    fX->Close();
    delete fY;
    delete fX;
}

// ============================================================================
// main
// ============================================================================

void build_cluster_x_y_and_draw_summary()
{
    Int_t oldLevel = gErrorIgnoreLevel;
    gErrorIgnoreLevel = kWarning;
    gStyle->SetOptStat(0);

    // -------------------------
    // X direction: TB07 TB06 TB10
    // -------------------------
    std::array<Dataset,3> dsX;

    //run 5 file name: TB02_All_events_Mar6_Run_5_135_227.root
    //run 3 file name: TB02_All_events_Mar6_Run_3_219_938.root
    //run 2 file name: TB02_All_events_Mar6_Run_2_128_213.root

    dsX[0].tag = "TB07";
    dsX[0].rootFileName = "TB07_All_events_Mar6_Run_6_230_310.root";
    dsX[0].txtFileName  = "TB07_ped_sigma_Mar6_Run_6.txt";
    dsX[0].txtOutName   = "TB07_Run6_cluster_info.txt";

    dsX[1].tag = "TB06";
    dsX[1].rootFileName = "TB06_All_events_Mar6_Run_6_230_310.root";
    dsX[1].txtFileName  = "TB06_ped_sigma_Mar6_Run_6.txt";
    dsX[1].txtOutName   = "TB06_Run6_cluster_info.txt";

    dsX[2].tag = "TB10";
    dsX[2].rootFileName = "TB10_All_events_Mar6_Run_6_230_310.root";
    dsX[2].txtFileName  = "TB10_ped_sigma_Mar6_Run_6.txt";
    dsX[2].txtOutName   = "TB10_Run6_cluster_info.txt";

    bool okX = ProcessGroup(dsX,
                            "cluster_building_07_06_10.root",
                            "Run6_TB07_TB06_TB10_allEvents_subtractPedCMN.pdf",
                            "Run6_TB07_TB06_TB10_allEvents_CMN.pdf",
                            "Run6_TB07_TB06_TB10_allEvents_cluster_check.pdf");

    if (!okX) {
        gErrorIgnoreLevel = oldLevel;
        return;
    }

    // -------------------------
    // Y direction: TB02 TB05 TB11
    // -------------------------
    std::array<Dataset,3> dsY;

    dsY[0].tag = "TB02";
    dsY[0].rootFileName = "TB02_All_events_Mar6_Run_6_230_310.root";
    dsY[0].txtFileName  = "TB02_ped_sigma_Mar6_Run_6.txt";
    dsY[0].txtOutName   = "TB02_Run6_cluster_info.txt";

    dsY[1].tag = "TB05";
    dsY[1].rootFileName = "TB05_All_events_Mar6_Run_6_230_310.root";
    dsY[1].txtFileName  = "TB05_ped_sigma_Mar6_Run_6.txt";
    dsY[1].txtOutName   = "TB05_Run6_cluster_info.txt";

    dsY[2].tag = "TB11";
    dsY[2].rootFileName = "TB11_All_events_Mar6_Run_6_230_310.root";
    dsY[2].txtFileName  = "TB11_ped_sigma_Mar6_Run_6.txt";
    dsY[2].txtOutName   = "TB11_Run6_cluster_info.txt";

    bool okY = ProcessGroup(dsY,
                            "cluster_building_02_05_11.root",
                            "Run6_TB02_TB05_TB11_allEvents_subtractPedCMN.pdf",
                            "Run6_TB02_TB05_TB11_allEvents_CMN.pdf",
                            "Run6_TB02_TB05_TB11_allEvents_cluster_check.pdf");

    if (!okY) {
        gErrorIgnoreLevel = oldLevel;
        return;
    }

    // -------------------------
    // summary plots
    // -------------------------
    DrawSummaryPlots("cluster_building_02_05_11.root", "cluster_building_07_06_10.root");

    gErrorIgnoreLevel = oldLevel;
}
