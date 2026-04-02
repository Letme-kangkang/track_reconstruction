#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TLine.h>
#include <TH1D.h>
#include <TH2F.h>
#include <TPad.h>
#include <TString.h>
#include <TAxis.h>

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>

struct ClusterCand {
    double channel = -999.0;
    double pos_cm  = -999.0;
    double value   = -999.0;
    int    size    = -1;
};

struct LayerEvent {
    int eventID = -1;
    std::vector<ClusterCand> clusters;
};

struct FitResult {
    double slope = 0.0;      // x = slope * z + intercept
    double intercept = 0.0;
    double chi2 = 1e30;
    int ndf = -1;
    bool valid = false;
};

struct ComboResult {
    FitResult fit;
    int idx[3] = {-1, -1, -1};
    double sumValue = -1.0;
    int sumSize = 999999;
    bool valid = false;
};

bool BetterCombo(const ComboResult& a, const ComboResult& b, double chi2Tol = 1e-12)
{
    if (!a.valid) return false;
    if (!b.valid) return true;

    if (a.fit.chi2 < b.fit.chi2 - chi2Tol) return true;
    if (a.fit.chi2 > b.fit.chi2 + chi2Tol) return false;

    if (a.sumValue > b.sumValue + 1e-9) return true;
    if (a.sumValue < b.sumValue - 1e-9) return false;

    if (a.sumSize < b.sumSize) return true;
    if (a.sumSize > b.sumSize) return false;

    return false;
}

double ComputeMidPlaneResidualUm(const std::array<double,3>& z,
                                 const std::array<double,3>& x,
                                 double* predMid_cm = nullptr)
{
    // 用 z[0] 和 z[2] 两点拉线，预测 z[1] 位置
    if (std::fabs(z[2] - z[0]) < 1e-15) {
        if (predMid_cm) *predMid_cm = -999.0;
        return -999.0;
    }

    const double pred =
        x[0] + (x[2] - x[0]) * (z[1] - z[0]) / (z[2] - z[0]);

    if (predMid_cm) *predMid_cm = pred;

    return (x[1] - pred) * 1.0e4; // cm -> um
}

FitResult FitStraightLine3(const std::array<double,3>& z,
                           const std::array<double,3>& x)
{
    FitResult res;

    const double S   = 3.0;
    const double Sz  = z[0] + z[1] + z[2];
    const double Szz = z[0]*z[0] + z[1]*z[1] + z[2]*z[2];
    const double Sx  = x[0] + x[1] + x[2];
    const double Szx = z[0]*x[0] + z[1]*x[1] + z[2]*x[2];

    const double denom = S * Szz - Sz * Sz;
    if (std::fabs(denom) < 1e-15) return res;

    res.slope     = (S * Szx - Sz * Sx) / denom;
    res.intercept = (Szz * Sx - Sz * Szx) / denom;

    res.chi2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double pred = res.slope * z[i] + res.intercept;
        const double r = x[i] - pred;
        res.chi2 += r * r;
    }

    res.ndf = 1;
    res.valid = true;
    return res;
}

std::map<int, LayerEvent> ReadLayerTree(TFile* fin,
                                        const std::string& treeName,
                                        double pitch_cm,
                                        double centerChannel,
                                        bool centerAtDetector = true)
{
    std::map<int, LayerEvent> out;

    TTree* tree = dynamic_cast<TTree*>(fin->Get(treeName.c_str()));
    if (!tree) {
        std::cerr << "ERROR: cannot find tree " << treeName << std::endl;
        return out;
    }

    int eventID = -1;
    int nClusters = 0;
    std::vector<double>* clusterPosition = nullptr;
    std::vector<int>*    clusterSize     = nullptr;
    std::vector<double>* clusterValue    = nullptr;

    tree->SetBranchAddress("eventID",         &eventID);
    tree->SetBranchAddress("nClusters",       &nClusters);
    tree->SetBranchAddress("clusterPosition", &clusterPosition);
    tree->SetBranchAddress("clusterSize",     &clusterSize);
    tree->SetBranchAddress("clusterValue",    &clusterValue);

    const Long64_t nEntries = tree->GetEntries();
    for (Long64_t ie = 0; ie < nEntries; ++ie) {
        tree->GetEntry(ie);

        LayerEvent evt;
        evt.eventID = eventID;

        int n = nClusters;
        if (clusterPosition) n = std::min(n, (int)clusterPosition->size());
        if (clusterSize)     n = std::min(n, (int)clusterSize->size());
        if (clusterValue)    n = std::min(n, (int)clusterValue->size());
        if (n < 0) n = 0;

        evt.clusters.reserve(n);
        for (int ic = 0; ic < n; ++ic) {
            ClusterCand c;
            c.channel = clusterPosition->at(ic);

            if (centerAtDetector)
                c.pos_cm = (clusterPosition->at(ic) - centerChannel) * pitch_cm;
            else
                c.pos_cm = clusterPosition->at(ic) * pitch_cm;

            c.size  = clusterSize->at(ic);
            c.value = clusterValue->at(ic);
            evt.clusters.push_back(c);
        }

        out[eventID] = evt;
    }

    std::cout << "Read tree " << treeName << " : " << out.size() << " events" << std::endl;
    return out;
}

ComboResult FindBestCombo(const LayerEvent& L0,
                          const LayerEvent& L1,
                          const LayerEvent& L2,
                          const std::array<double,3>& z)
{
    ComboResult best;

    if (L0.clusters.empty() || L1.clusters.empty() || L2.clusters.empty()) {
        return best;
    }

    for (int i = 0; i < (int)L0.clusters.size(); ++i) {
        for (int j = 0; j < (int)L1.clusters.size(); ++j) {
            for (int k = 0; k < (int)L2.clusters.size(); ++k) {
                std::array<double,3> x = {
                    L0.clusters[i].pos_cm,
                    L1.clusters[j].pos_cm,
                    L2.clusters[k].pos_cm
                };

                ComboResult cand;
                cand.fit = FitStraightLine3(z, x);
                cand.idx[0] = i;
                cand.idx[1] = j;
                cand.idx[2] = k;
                cand.sumValue = L0.clusters[i].value
                              + L1.clusters[j].value
                              + L2.clusters[k].value;
                cand.sumSize = L0.clusters[i].size
                             + L1.clusters[j].size
                             + L2.clusters[k].size;
                cand.valid = cand.fit.valid;

                if (BetterCombo(cand, best)) best = cand;
            }
        }
    }

    return best;
}

void CollectAllPoints(const LayerEvent& L0,
                      const LayerEvent& L1,
                      const LayerEvent& L2,
                      const std::array<double,3>& z,
                      std::vector<double>& posAll,
                      std::vector<double>& zAll)
{
    posAll.clear();
    zAll.clear();

    for (const auto& c : L0.clusters) {
        posAll.push_back(c.pos_cm);
        zAll.push_back(z[0]);
    }
    for (const auto& c : L1.clusters) {
        posAll.push_back(c.pos_cm);
        zAll.push_back(z[1]);
    }
    for (const auto& c : L2.clusters) {
        posAll.push_back(c.pos_cm);
        zAll.push_back(z[2]);
    }
}

void GetSelectedPoints(const LayerEvent& L0,
                       const LayerEvent& L1,
                       const LayerEvent& L2,
                       const ComboResult& best,
                       const std::array<double,3>& z,
                       std::vector<double>& posSel,
                       std::vector<double>& zSel,
                       std::vector<double>& chSel,
                       std::vector<double>& valSel,
                       std::vector<int>& sizeSel)
{
    posSel.clear();
    zSel.clear();
    chSel.clear();
    valSel.clear();
    sizeSel.clear();

    if (!best.valid) return;

    const ClusterCand& c0 = L0.clusters[best.idx[0]];
    const ClusterCand& c1 = L1.clusters[best.idx[1]];
    const ClusterCand& c2 = L2.clusters[best.idx[2]];

    posSel.push_back(c0.pos_cm); zSel.push_back(z[0]); chSel.push_back(c0.channel); valSel.push_back(c0.value); sizeSel.push_back(c0.size);
    posSel.push_back(c1.pos_cm); zSel.push_back(z[1]); chSel.push_back(c1.channel); valSel.push_back(c1.value); sizeSel.push_back(c1.size);
    posSel.push_back(c2.pos_cm); zSel.push_back(z[2]); chSel.push_back(c2.channel); valSel.push_back(c2.value); sizeSel.push_back(c2.size);
}

void DetermineXRange(const std::vector<double>& posAll,
                     const std::vector<double>& posSel,
                     double& xmin,
                     double& xmax)
{
    double mn =  std::numeric_limits<double>::max();
    double mx = -std::numeric_limits<double>::max();

    for (double v : posAll) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    for (double v : posSel) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }

    if (mn > mx) {
        mn = -6.0;
        mx =  6.0;
    }

    double width = mx - mn;
    if (width < 0.3) width = 0.3;

    xmin = mn - 0.18 * width - 0.05;
    xmax = mx + 0.18 * width + 0.05;
}

TH2F* MakeFrame(const char* name,
                const char* title,
                double xmin, double xmax,
                double ymin, double ymax)
{
    TH2F* h = new TH2F(name, title, 100, xmin, xmax, 100, ymin, ymax);
    h->SetDirectory(nullptr);
    h->GetXaxis()->SetTitleSize(0.042);
    h->GetYaxis()->SetTitleSize(0.042);
    h->GetXaxis()->SetLabelSize(0.034);
    h->GetYaxis()->SetLabelSize(0.034);
    h->GetXaxis()->SetTitleOffset(1.00);
    h->GetYaxis()->SetTitleOffset(1.15);
    h->Draw();
    return h;
}

void DrawProjection(TPad* pad,
                    const char* projTitle,
                    const char* axisLabel,
                    const std::array<std::string,3>& layerNames,
                    const LayerEvent& L0,
                    const LayerEvent& L1,
                    const LayerEvent& L2,
                    const ComboResult& best,
                    const std::array<double,3>& z,
                    const char* formulaVar,
                    int eventID)
{
    pad->cd();
    pad->Clear();
    pad->SetGrid();
    pad->SetLeftMargin(0.12);
    pad->SetRightMargin(0.04);
    pad->SetTopMargin(0.08);
    pad->SetBottomMargin(0.12);

    std::vector<double> posAll, zAll;
    std::vector<double> posSel, zSel, chSel, valSel;
    std::vector<int> sizeSel;

    CollectAllPoints(L0, L1, L2, z, posAll, zAll);
    GetSelectedPoints(L0, L1, L2, best, z, posSel, zSel, chSel, valSel, sizeSel);

    double xmin, xmax;
    DetermineXRange(posAll, posSel, xmin, xmax);

    static int frameCounter = 0;
    TH2F* frame = MakeFrame(Form("hframe_evt%d_%s_%d", eventID, formulaVar, frameCounter++),
                            Form("%s;%s [cm];z [cm]", projTitle, axisLabel),
                            xmin, xmax, -0.4, 6.4);

    TGraph* gAll = nullptr;
    if (!posAll.empty()) {
        gAll = new TGraph((int)posAll.size(), posAll.data(), zAll.data());
        gAll->SetMarkerStyle(24);
        gAll->SetMarkerSize(1.00);
        gAll->SetMarkerColor(kGray + 2);
        gAll->Draw("P SAME");
    }

    TGraph* gSel = nullptr;
    if (!posSel.empty()) {
        gSel = new TGraph((int)posSel.size(), posSel.data(), zSel.data());
        gSel->SetMarkerStyle(20);
        gSel->SetMarkerSize(1.10);
        gSel->SetMarkerColor(kRed + 1);
        gSel->Draw("P SAME");
    }

    TLine* fitLine = nullptr;
    TLine* refLine = nullptr;
    TLine* resLine = nullptr;

    double predMid_cm = -999.0;
    double residual_um = -999.0;

    if (best.valid && posSel.size() == 3) {
        // 原来的三点拟合线
        const double zmin = 0.0;
        const double zmax = 6.0;
        const double x1 = best.fit.slope * zmin + best.fit.intercept;
        const double x2 = best.fit.slope * zmax + best.fit.intercept;

        fitLine = new TLine(x1, zmin, x2, zmax);
        fitLine->SetLineColor(kBlue + 1);
        fitLine->SetLineWidth(2);
        fitLine->Draw("SAME");

        // 用 z=6 和 z=0 两点拉参考直线
        refLine = new TLine(posSel[2], z[2], posSel[0], z[0]);
        refLine->SetLineColor(kGreen + 2);
        refLine->SetLineStyle(2);
        refLine->SetLineWidth(2);
        refLine->Draw("SAME");

        std::array<double,3> xSel = {posSel[0], posSel[1], posSel[2]};
        residual_um = ComputeMidPlaneResidualUm(z, xSel, &predMid_cm);

        // 在 z=3 cm 平面上，把“实测点”和“参考线预测点”连起来
        resLine = new TLine(posSel[1], z[1], predMid_cm, z[1]);
        resLine->SetLineColor(kMagenta + 1);
        resLine->SetLineStyle(7);
        resLine->SetLineWidth(3);
        resLine->Draw("SAME");
    }

    TLegend* leg = new TLegend(0.64, 0.76, 0.94, 0.90);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.027);
    if (gAll)    leg->AddEntry(gAll, "All clusters", "p");
    if (gSel)    leg->AddEntry(gSel, "Selected clusters", "p");
    if (fitLine) leg->AddEntry(fitLine, "3-point fit", "l");
    if (refLine) leg->AddEntry(refLine, "Line from z=6 & 0", "l");
    if (resLine) leg->AddEntry(resLine, "Residual at z=3", "l");
    leg->Draw();

    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.025);
    latex.SetTextFont(42);

    latex.DrawLatex(0.14, 0.905,
        Form("Layers: %s(z=%.0f), %s(z=%.0f), %s(z=%.0f)",
             layerNames[0].c_str(), z[0],
             layerNames[1].c_str(), z[1],
             layerNames[2].c_str(), z[2]));

    latex.DrawLatex(0.14, 0.857,
        Form("nClusters = [%d, %d, %d]",
             (int)L0.clusters.size(), (int)L1.clusters.size(), (int)L2.clusters.size()));

    if (best.valid) {
        latex.DrawLatex(0.14, 0.809,
            Form("selected ch = [%.2f, %.2f, %.2f]",
                 chSel[0], chSel[1], chSel[2]));

        latex.DrawLatex(0.14, 0.761,
            Form("selected value = [%.2f, %.2f, %.2f]",
                 valSel[0], valSel[1], valSel[2]));

        latex.DrawLatex(0.14, 0.713,
            Form("selected size = [%d, %d, %d]",
                 sizeSel[0], sizeSel[1], sizeSel[2]));

        latex.DrawLatex(0.14, 0.665,
            Form("#chi^{2} = %.5g", best.fit.chi2));

        latex.DrawLatex(0.14, 0.617,
            Form("%s(z) = %.5g #times z %+.5g",
                 formulaVar, best.fit.slope, best.fit.intercept));

        latex.DrawLatex(0.14, 0.569,
            Form("%s_{pred}(z=3 cm, from z=6&0) = %.6f cm",
                 formulaVar, predMid_cm));

        latex.DrawLatex(0.14, 0.521,
            Form("Residual at z=3 cm = %.1f #mum", residual_um));
    } else {
        latex.DrawLatex(0.14, 0.761, "No valid 3-layer combo");
    }

    TLatex lz;
    lz.SetTextSize(0.023);
    lz.SetTextFont(62);

    const double xText = xmin + 0.03 * (xmax - xmin);
    lz.DrawLatex(xText, z[0] + 0.08, layerNames[0].c_str());
    lz.DrawLatex(xText, z[1] + 0.08, layerNames[1].c_str());
    lz.DrawLatex(xText, z[2] + 0.08, layerNames[2].c_str());

    pad->Modified();
    pad->Update();

    (void)frame;
}

void DrawOneEventDisplay(TCanvas* c,
                         int eventID,
                         bool validTrack,
                         double theta,
                         double phi,
                         double chi2,
                         const LayerEvent& xTopEvt,
                         const LayerEvent& xMidEvt,
                         const LayerEvent& xBotEvt,
                         const LayerEvent& yTopEvt,
                         const LayerEvent& yMidEvt,
                         const LayerEvent& yBotEvt,
                         const ComboResult& bestX,
                         const ComboResult& bestY,
                         const std::array<double,3>& z_cm,
                         const std::string& pdfName,
                         bool firstPage)
{
    c->Clear();
    c->Divide(2,1,0.01,0.01);

    TPad* p1 = (TPad*)c->cd(1);
    DrawProjection(p1,
                   "X-Z projection",
                   "x",
                   {std::string("TB07"), std::string("TB06"), std::string("TB10")},
                   xTopEvt, xMidEvt, xBotEvt, bestX, z_cm, "x", eventID);

    TPad* p2 = (TPad*)c->cd(2);
    DrawProjection(p2,
                   "Y-Z projection",
                   "y",
                   {std::string("TB02"), std::string("TB05"), std::string("TB11")},
                   yTopEvt, yMidEvt, yBotEvt, bestY, z_cm, "y", eventID);

    c->cd();
    TLatex top;
    top.SetNDC();
    top.SetTextSize(0.022);
    top.SetTextFont(42);

    if (validTrack) {
        top.DrawLatex(0.22, 0.985,
            Form("Event %d   |   validTrack = 1   |   #theta = %.5f rad   |   #phi = %.5f rad   |   #chi^{2}_{tot} = %.5g",
                 eventID, theta, phi, chi2));
    } else {
        top.DrawLatex(0.40, 0.985,
            Form("Event %d   |   validTrack = 0", eventID));
    }

    c->Update();

    if (firstPage) c->Print((pdfName + "[").c_str());
    c->Print(pdfName.c_str());
}

void DrawHistWithSummary(TPad* pad, TH1D* h, int color, const char* extraLine = "")
{
    pad->cd();
    pad->SetGrid();
    pad->SetLeftMargin(0.12);
    pad->SetRightMargin(0.05);
    pad->SetTopMargin(0.08);
    pad->SetBottomMargin(0.12);

    h->SetLineColor(color);
    h->SetLineWidth(2);
    h->SetFillStyle(0);
    h->GetXaxis()->SetTitleSize(0.045);
    h->GetYaxis()->SetTitleSize(0.045);
    h->GetXaxis()->SetLabelSize(0.038);
    h->GetYaxis()->SetLabelSize(0.038);
    h->GetXaxis()->SetTitleOffset(1.05);
    h->GetYaxis()->SetTitleOffset(1.18);
    h->Draw("HIST");

    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(42);
    latex.SetTextSize(0.040);
    latex.DrawLatex(0.16, 0.88, Form("Entries = %.0f", h->GetEntries()));
    latex.DrawLatex(0.16, 0.81, Form("Mean = %.3f", h->GetMean()));
    latex.DrawLatex(0.16, 0.74, Form("RMS = %.3f", h->GetRMS()));
    if (std::string(extraLine).size() > 0) {
        latex.DrawLatex(0.16, 0.67, extraLine);
    }
}

void MakeEfficiencyPdf(const std::string& pdfName,
                       long long nTotal,
                       long long nAll6,
                       long long nGoodTrack,
                       TFile* fout)
{
    const double eff_total = (nTotal > 0) ? (double)nGoodTrack / (double)nTotal : 0.0;
    const double eff_all6  = (nAll6  > 0) ? (double)nGoodTrack / (double)nAll6  : 0.0;

    TCanvas* cEff = new TCanvas("cEff_trackReco", "Track reconstruction efficiency", 1400, 700);
    cEff->Divide(2,1);

    gStyle->SetPaintTextFormat(".4f");

    // 左边：数量统计
    cEff->cd(1);
    gPad->SetGrid();
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);
    gPad->SetTopMargin(0.08);
    gPad->SetBottomMargin(0.14);

    TH1D* h_counts = new TH1D("h_trackRecoCounts",
                              "Track reconstruction counts;Category;Events",
                              3, 0.5, 3.5);
    h_counts->SetDirectory(nullptr);
    h_counts->SetBinContent(1, nTotal);
    h_counts->SetBinContent(2, nAll6);
    h_counts->SetBinContent(3, nGoodTrack);
    h_counts->GetXaxis()->SetBinLabel(1, "Total events");
    h_counts->GetXaxis()->SetBinLabel(2, "All 6 layers");
    h_counts->GetXaxis()->SetBinLabel(3, "Valid tracks");
    h_counts->SetFillColor(kAzure - 9);
    h_counts->SetLineColor(kAzure + 2);
    h_counts->SetLineWidth(2);

    double ymax_counts = (double)std::max(std::max(nTotal, nAll6), nGoodTrack);
    if (ymax_counts < 1.0) ymax_counts = 1.0;
    h_counts->SetMaximum(1.25 * ymax_counts);
    h_counts->GetXaxis()->SetTitleSize(0.045);
    h_counts->GetYaxis()->SetTitleSize(0.045);
    h_counts->GetXaxis()->SetLabelSize(0.040);
    h_counts->GetYaxis()->SetLabelSize(0.040);
    h_counts->Draw("HIST TEXT0");

    // 右边：效率
    cEff->cd(2);
    gPad->SetGrid();
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.05);
    gPad->SetTopMargin(0.08);
    gPad->SetBottomMargin(0.14);

    TH1D* h_eff = new TH1D("h_trackRecoEfficiency",
                           "Track reconstruction efficiency;Category;Efficiency",
                           2, 0.5, 2.5);
    h_eff->SetDirectory(nullptr);
    h_eff->SetBinContent(1, eff_total);
    h_eff->SetBinContent(2, eff_all6);
    h_eff->GetXaxis()->SetBinLabel(1, "valid / total");
    h_eff->GetXaxis()->SetBinLabel(2, "valid / all6");
    h_eff->SetFillColor(kGreen - 8);
    h_eff->SetLineColor(kGreen + 2);
    h_eff->SetLineWidth(2);
    h_eff->SetMinimum(0.0);
    h_eff->SetMaximum(1.05);
    h_eff->GetXaxis()->SetTitleSize(0.045);
    h_eff->GetYaxis()->SetTitleSize(0.045);
    h_eff->GetXaxis()->SetLabelSize(0.040);
    h_eff->GetYaxis()->SetLabelSize(0.040);
    h_eff->Draw("HIST TEXT0");

    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(42);
    latex.SetTextSize(0.040);
    latex.DrawLatex(0.16, 0.88,
        Form("validTrack / totalEvents = %lld / %lld = %.6f",
             nGoodTrack, nTotal, eff_total));
    latex.DrawLatex(0.16, 0.80,
        Form("validTrack / all6Layers  = %lld / %lld = %.6f",
             nGoodTrack, nAll6, eff_all6));

    cEff->Update();
    cEff->Print(pdfName.c_str());

    if (fout) {
        fout->cd();
        h_counts->Write();
        h_eff->Write();
        cEff->Write();
    }
}

void MakeResidualPdf(const std::string& pdfName,
                     TH1D* h_residualX_um,
                     TH1D* h_residualY_um,
                     TH1D* h_residualR_um,
                     TFile* fout)
{
    TCanvas* cRes = new TCanvas("cResidual_trackReco", "Track residual distributions", 1500, 950);
    cRes->Divide(2,2);

    cRes->cd(1);
    DrawHistWithSummary((TPad*)gPad, h_residualX_um, kBlue + 1, "mid-plane x residual");

    cRes->cd(2);
    DrawHistWithSummary((TPad*)gPad, h_residualY_um, kRed + 1, "mid-plane y residual");

    cRes->cd(3);
    DrawHistWithSummary((TPad*)gPad, h_residualR_um, kMagenta + 1, "transverse residual");

    cRes->cd(4);
    gPad->SetLeftMargin(0.08);
    gPad->SetRightMargin(0.04);
    gPad->SetTopMargin(0.06);
    gPad->SetBottomMargin(0.08);

    TLatex latex;
    latex.SetNDC();
    latex.SetTextFont(42);

    latex.SetTextSize(0.055);
    latex.DrawLatex(0.08, 0.90, "Residual summary");

    latex.SetTextSize(0.042);
    latex.DrawLatex(0.08, 0.78,
        Form("X residual: Entries = %.0f, Mean = %.3f #mum, RMS = %.3f #mum",
             h_residualX_um->GetEntries(), h_residualX_um->GetMean(), h_residualX_um->GetRMS()));

    latex.DrawLatex(0.08, 0.66,
        Form("Y residual: Entries = %.0f, Mean = %.3f #mum, RMS = %.3f #mum",
             h_residualY_um->GetEntries(), h_residualY_um->GetMean(), h_residualY_um->GetRMS()));

    latex.DrawLatex(0.08, 0.54,
        Form("R residual: Entries = %.0f, Mean = %.3f #mum, RMS = %.3f #mum",
             h_residualR_um->GetEntries(), h_residualR_um->GetMean(), h_residualR_um->GetRMS()));

    latex.SetTextSize(0.038);
    latex.DrawLatex(0.08, 0.38, "Definitions:");
    latex.DrawLatex(0.10, 0.28, "#Deltax(z=3 cm) = x_{meas} - x_{pred}(from z=6 and 0)");
    latex.DrawLatex(0.10, 0.20, "#Deltay(z=3 cm) = y_{meas} - y_{pred}(from z=6 and 0)");
    latex.DrawLatex(0.10, 0.12, "#Deltar = #sqrt{#Deltax^{2} + #Deltay^{2}}");

    cRes->Update();
    cRes->Print(pdfName.c_str());

    if (fout) {
        fout->cd();
        cRes->Write();
    }
}

void track_reco_eventDisplay(int maxEvents = -1, bool drawOnlyValidTrack = true)
{
    gStyle->SetOptStat(0);
    gStyle->SetTitleFontSize(0.04);

    const double pitch_cm = 109e-4;
    const double centerChannel = 511.5;
    const bool centerAtDetector = true;

    // 正确的 z 几何：
    // top: TB07/TB02 -> 6 cm
    // mid: TB06/TB05 -> 3 cm
    // bottom: TB10/TB11 -> 0 cm
    const std::array<double,3> z_cm = {6.0, 3.0, 0.0};

    TFile* fx = TFile::Open("cluster_building_07_06_10.root", "READ");
    TFile* fy = TFile::Open("cluster_building_02_05_11.root", "READ");

    if (!fx || fx->IsZombie()) {
        std::cerr << "ERROR: cannot open cluster_building_07_06_10.root" << std::endl;
        return;
    }
    if (!fy || fy->IsZombie()) {
        std::cerr << "ERROR: cannot open cluster_building_02_05_11.root" << std::endl;
        if (fx) fx->Close();
        return;
    }

    auto xTop = ReadLayerTree(fx, "clusterTree_TB07", pitch_cm, centerChannel, centerAtDetector);
    auto xMid = ReadLayerTree(fx, "clusterTree_TB06", pitch_cm, centerChannel, centerAtDetector);
    auto xBot = ReadLayerTree(fx, "clusterTree_TB10", pitch_cm, centerChannel, centerAtDetector);

    auto yTop = ReadLayerTree(fy, "clusterTree_TB02", pitch_cm, centerChannel, centerAtDetector);
    auto yMid = ReadLayerTree(fy, "clusterTree_TB05", pitch_cm, centerChannel, centerAtDetector);
    auto yBot = ReadLayerTree(fy, "clusterTree_TB11", pitch_cm, centerChannel, centerAtDetector);

    std::set<int> allEventIDs;
    for (const auto& kv : xTop) allEventIDs.insert(kv.first);
    for (const auto& kv : xMid) allEventIDs.insert(kv.first);
    for (const auto& kv : xBot) allEventIDs.insert(kv.first);
    for (const auto& kv : yTop) allEventIDs.insert(kv.first);
    for (const auto& kv : yMid) allEventIDs.insert(kv.first);
    for (const auto& kv : yBot) allEventIDs.insert(kv.first);

    TFile* fout = TFile::Open("track_reco_eventDisplay.root", "RECREATE");
    if (!fout || fout->IsZombie()) {
        std::cerr << "ERROR: cannot create output file track_reco_eventDisplay.root" << std::endl;
        fx->Close();
        fy->Close();
        return;
    }

    TTree* outTree = new TTree("trackTree", "Reconstructed cosmic muon tracks with event display");

    int out_eventID = -1;
    int out_hasAll6Layers = 0;
    int out_validX = 0;
    int out_validY = 0;
    int out_validTrack = 0;

    double out_xPos[3] = {-999., -999., -999.};
    double out_yPos[3] = {-999., -999., -999.};
    double out_xChannel[3] = {-999., -999., -999.};
    double out_yChannel[3] = {-999., -999., -999.};
    double out_xValue[3] = {-999., -999., -999.};
    double out_yValue[3] = {-999., -999., -999.};
    int    out_xSize[3] = {-1, -1, -1};
    int    out_ySize[3] = {-1, -1, -1};

    int out_xClusterIndex[3] = {-1, -1, -1};
    int out_yClusterIndex[3] = {-1, -1, -1};

    int out_nClustersX[3] = {0, 0, 0};
    int out_nClustersY[3] = {0, 0, 0};

    double out_ax = -999., out_bx = -999., out_chi2x = -999.;
    double out_ay = -999., out_by = -999., out_chi2y = -999.;
    double out_chi2 = -999.;
    double out_theta = -999.;
    double out_phi   = -999.;

    double out_residualX_um = -999.;
    double out_residualY_um = -999.;
    double out_residualR_um = -999.;

    outTree->Branch("eventID",        &out_eventID,      "eventID/I");
    outTree->Branch("hasAll6Layers",  &out_hasAll6Layers,"hasAll6Layers/I");
    outTree->Branch("validX",         &out_validX,       "validX/I");
    outTree->Branch("validY",         &out_validY,       "validY/I");
    outTree->Branch("validTrack",     &out_validTrack,   "validTrack/I");

    outTree->Branch("xPos",           out_xPos,          "xPos[3]/D");
    outTree->Branch("yPos",           out_yPos,          "yPos[3]/D");
    outTree->Branch("xChannel",       out_xChannel,      "xChannel[3]/D");
    outTree->Branch("yChannel",       out_yChannel,      "yChannel[3]/D");
    outTree->Branch("xValue",         out_xValue,        "xValue[3]/D");
    outTree->Branch("yValue",         out_yValue,        "yValue[3]/D");
    outTree->Branch("xSize",          out_xSize,         "xSize[3]/I");
    outTree->Branch("ySize",          out_ySize,         "ySize[3]/I");

    outTree->Branch("xClusterIndex",  out_xClusterIndex, "xClusterIndex[3]/I");
    outTree->Branch("yClusterIndex",  out_yClusterIndex, "yClusterIndex[3]/I");
    outTree->Branch("nClustersX",     out_nClustersX,    "nClustersX[3]/I");
    outTree->Branch("nClustersY",     out_nClustersY,    "nClustersY[3]/I");

    outTree->Branch("ax",             &out_ax,           "ax/D");
    outTree->Branch("bx",             &out_bx,           "bx/D");
    outTree->Branch("chi2x",          &out_chi2x,        "chi2x/D");
    outTree->Branch("ay",             &out_ay,           "ay/D");
    outTree->Branch("by",             &out_by,           "by/D");
    outTree->Branch("chi2y",          &out_chi2y,        "chi2y/D");
    outTree->Branch("chi2",           &out_chi2,         "chi2/D");
    outTree->Branch("theta",          &out_theta,        "theta/D");
    outTree->Branch("phi",            &out_phi,          "phi/D");

    outTree->Branch("residualX_um",   &out_residualX_um, "residualX_um/D");
    outTree->Branch("residualY_um",   &out_residualY_um, "residualY_um/D");
    outTree->Branch("residualR_um",   &out_residualR_um, "residualR_um/D");

    TH1D* h_chi2x = new TH1D("h_chi2x", "Best x-fit #chi^{2};#chi^{2};Events", 300, 0, 0.5);
    TH1D* h_chi2y = new TH1D("h_chi2y", "Best y-fit #chi^{2};#chi^{2};Events", 300, 0, 0.5);
    TH1D* h_theta = new TH1D("h_theta", "Track #theta;#theta [rad];Events", 200, 0, 0.5);
    TH1D* h_phi   = new TH1D("h_phi",   "Track #phi;#phi [rad];Events", 200, -3.2, 3.2);

    TH1D* h_residualX_um = new TH1D("h_residualX_um",
        "Mid-plane residual in x;#Deltax(z=3 cm) [#mum];Events", 400, -2000, 2000);

    TH1D* h_residualY_um = new TH1D("h_residualY_um",
        "Mid-plane residual in y;#Deltay(z=3 cm) [#mum];Events", 400, -2000, 2000);

    TH1D* h_residualR_um = new TH1D("h_residualR_um",
        "Transverse residual at z=3 cm;#sqrt{#Deltax^{2}+#Deltay^{2}} [#mum];Events", 400, 0, 3000);

    const std::string pdfEventDisplay = "track_event_display.pdf";
    const std::string pdfEfficiency   = "track_reco_efficiency.pdf";
    const std::string pdfResidual     = "track_residual_distributions.pdf";

    TCanvas* cDisplay = new TCanvas("cDisplay", "Track Event Display", 1600, 700);
    bool pdfOpened = false;
    int nPagesDrawn = 0;

    long long nTotal = 0;
    long long nAll6 = 0;
    long long nGoodTrack = 0;

    for (int eventID : allEventIDs) {
        ++nTotal;

        out_eventID = eventID;
        out_hasAll6Layers = 0;
        out_validX = 0;
        out_validY = 0;
        out_validTrack = 0;

        for (int i = 0; i < 3; ++i) {
            out_xPos[i] = out_yPos[i] = -999.;
            out_xChannel[i] = out_yChannel[i] = -999.;
            out_xValue[i] = out_yValue[i] = -999.;
            out_xSize[i] = out_ySize[i] = -1;
            out_xClusterIndex[i] = out_yClusterIndex[i] = -1;
            out_nClustersX[i] = out_nClustersY[i] = 0;
        }

        out_ax = out_bx = out_chi2x = -999.;
        out_ay = out_by = out_chi2y = -999.;
        out_chi2 = out_theta = out_phi = -999.;
        out_residualX_um = out_residualY_um = out_residualR_um = -999.;

        auto itx0 = xTop.find(eventID);
        auto itx1 = xMid.find(eventID);
        auto itx2 = xBot.find(eventID);

        auto ity0 = yTop.find(eventID);
        auto ity1 = yMid.find(eventID);
        auto ity2 = yBot.find(eventID);

        if (itx0 == xTop.end() || itx1 == xMid.end() || itx2 == xBot.end() ||
            ity0 == yTop.end() || ity1 == yMid.end() || ity2 == yBot.end()) {
            outTree->Fill();
            continue;
        }

        out_hasAll6Layers = 1;
        ++nAll6;

        const LayerEvent& ex0 = itx0->second;
        const LayerEvent& ex1 = itx1->second;
        const LayerEvent& ex2 = itx2->second;

        const LayerEvent& ey0 = ity0->second;
        const LayerEvent& ey1 = ity1->second;
        const LayerEvent& ey2 = ity2->second;

        out_nClustersX[0] = (int)ex0.clusters.size();
        out_nClustersX[1] = (int)ex1.clusters.size();
        out_nClustersX[2] = (int)ex2.clusters.size();

        out_nClustersY[0] = (int)ey0.clusters.size();
        out_nClustersY[1] = (int)ey1.clusters.size();
        out_nClustersY[2] = (int)ey2.clusters.size();

        ComboResult bestX = FindBestCombo(ex0, ex1, ex2, z_cm);
        ComboResult bestY = FindBestCombo(ey0, ey1, ey2, z_cm);

        if (bestX.valid) {
            out_validX = 1;
            out_ax = bestX.fit.slope;
            out_bx = bestX.fit.intercept;
            out_chi2x = bestX.fit.chi2;

            for (int i = 0; i < 3; ++i) out_xClusterIndex[i] = bestX.idx[i];

            const ClusterCand& c0 = ex0.clusters[bestX.idx[0]];
            const ClusterCand& c1 = ex1.clusters[bestX.idx[1]];
            const ClusterCand& c2 = ex2.clusters[bestX.idx[2]];

            out_xPos[0] = c0.pos_cm; out_xPos[1] = c1.pos_cm; out_xPos[2] = c2.pos_cm;
            {
                std::array<double,3> xSel = {out_xPos[0], out_xPos[1], out_xPos[2]};
                out_residualX_um = ComputeMidPlaneResidualUm(z_cm, xSel);
                h_residualX_um->Fill(out_residualX_um);
            }
            out_xChannel[0] = c0.channel; out_xChannel[1] = c1.channel; out_xChannel[2] = c2.channel;
            out_xValue[0] = c0.value; out_xValue[1] = c1.value; out_xValue[2] = c2.value;
            out_xSize[0] = c0.size; out_xSize[1] = c1.size; out_xSize[2] = c2.size;

            h_chi2x->Fill(out_chi2x);
        }

        if (bestY.valid) {
            out_validY = 1;
            out_ay = bestY.fit.slope;
            out_by = bestY.fit.intercept;
            out_chi2y = bestY.fit.chi2;

            for (int i = 0; i < 3; ++i) out_yClusterIndex[i] = bestY.idx[i];

            const ClusterCand& c0 = ey0.clusters[bestY.idx[0]];
            const ClusterCand& c1 = ey1.clusters[bestY.idx[1]];
            const ClusterCand& c2 = ey2.clusters[bestY.idx[2]];

            out_yPos[0] = c0.pos_cm; out_yPos[1] = c1.pos_cm; out_yPos[2] = c2.pos_cm;
            {
                std::array<double,3> ySel = {out_yPos[0], out_yPos[1], out_yPos[2]};
                out_residualY_um = ComputeMidPlaneResidualUm(z_cm, ySel);
                h_residualY_um->Fill(out_residualY_um);
            }
            out_yChannel[0] = c0.channel; out_yChannel[1] = c1.channel; out_yChannel[2] = c2.channel;
            out_yValue[0] = c0.value; out_yValue[1] = c1.value; out_yValue[2] = c2.value;
            out_ySize[0] = c0.size; out_ySize[1] = c1.size; out_ySize[2] = c2.size;

            h_chi2y->Fill(out_chi2y);
        }

        if (out_validX && out_validY) {
            out_validTrack = 1;
            out_chi2 = out_chi2x + out_chi2y;

            const double tx = out_ax;
            const double ty = out_ay;

            out_theta = std::atan(std::sqrt(tx*tx + ty*ty));
            out_phi   = std::atan2(ty, tx);

            out_residualR_um = std::hypot(out_residualX_um, out_residualY_um);

            h_theta->Fill(out_theta);
            h_phi->Fill(out_phi);
            h_residualR_um->Fill(out_residualR_um);
            ++nGoodTrack;
        }

        outTree->Fill();

        bool shouldDraw = true;
        if (drawOnlyValidTrack && !out_validTrack) shouldDraw = false;
        if (maxEvents > 0 && nPagesDrawn >= maxEvents) shouldDraw = false;

        if (shouldDraw) {
            const bool firstPage = !pdfOpened;
            if (!pdfOpened) pdfOpened = true;

            DrawOneEventDisplay(cDisplay,
                                out_eventID,
                                out_validTrack,
                                out_theta,
                                out_phi,
                                out_chi2,
                                ex0, ex1, ex2,
                                ey0, ey1, ey2,
                                bestX, bestY,
                                z_cm,
                                pdfEventDisplay,
                                firstPage);

            ++nPagesDrawn;
        }

        if (maxEvents > 0 && nPagesDrawn >= maxEvents) {
            break;
        }
    }

    if (pdfOpened) {
        cDisplay->Print((pdfEventDisplay + "]").c_str());
    }

    fout->cd();
    outTree->Write();
    h_chi2x->Write();
    h_chi2y->Write();
    h_theta->Write();
    h_phi->Write();
    h_residualX_um->Write();
    h_residualY_um->Write();
    h_residualR_um->Write();

    // QA canvas
    TCanvas* cQA = new TCanvas("cQA", "Track QA", 1500, 900);
    cQA->Divide(3,2);
    cQA->cd(1); h_chi2x->Draw();
    cQA->cd(2); h_chi2y->Draw();
    cQA->cd(3); h_theta->Draw();
    cQA->cd(4); h_phi->Draw();
    cQA->cd(5); h_residualX_um->Draw();
    cQA->cd(6); h_residualR_um->Draw();
    cQA->Write();

    // 新增两个 PDF
    MakeEfficiencyPdf(pdfEfficiency, nTotal, nAll6, nGoodTrack, fout);
    MakeResidualPdf(pdfResidual, h_residualX_um, h_residualY_um, h_residualR_um, fout);

    fout->Close();
    fx->Close();
    fy->Close();

    const double eff_total = (nTotal > 0) ? (double)nGoodTrack / (double)nTotal : 0.0;
    const double eff_all6  = (nAll6  > 0) ? (double)nGoodTrack / (double)nAll6  : 0.0;

    std::cout << "\n========== Track reconstruction + event display summary ==========\n";
    std::cout << "Total unique events         : " << nTotal << "\n";
    std::cout << "Events with all 6 layers    : " << nAll6 << "\n";
    std::cout << "Events with valid 3D track  : " << nGoodTrack << "\n";
    std::cout << "Reco efficiency (valid/all) : " << eff_total << "\n";
    std::cout << "Reco efficiency (valid/all6): " << eff_all6  << "\n";
    std::cout << "Pages drawn to PDF          : " << nPagesDrawn << "\n";
    std::cout << "Output ROOT file            : track_reco_eventDisplay.root\n";
    std::cout << "Output PDF file             : " << pdfEventDisplay << "\n";
    std::cout << "Efficiency PDF file         : " << pdfEfficiency   << "\n";
    std::cout << "Residual PDF file           : " << pdfResidual     << "\n";
    std::cout << "==================================================================\n" << std::endl;
}
