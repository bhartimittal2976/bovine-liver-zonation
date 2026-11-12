#!/usr/bin/env bash
# Download and convert the cross-species zonation references used by 03_cross_species_zonation.py.
# Needs curl, unzip and a Python with pandas + openpyxl (the spatial_mac env has no openpyxl;
# a throwaway venv works: python3 -m venv /tmp/x && /tmp/x/bin/pip install pandas openpyxl).
set -euo pipefail
PY=${PY:-python3}
R="$(cd "$(dirname "$0")/.." && pwd)/data/reference"
mkdir -p "$R"/{halpern2017,brosch2018,yakubovsky2026,orthologs,csv}
cd "$R"

EPMC=https://www.ebi.ac.uk/europepmc/webservices/rest
curl -sL -o halpern2017/supp.zip    "$EPMC/PMC5321580/supplementaryFiles"    # Halpern 2017 Nature
curl -sL -o brosch2018/supp.zip     "$EPMC/PMC6175862/supplementaryFiles"    # Brosch 2018 Nat Commun
curl -sL -o yakubovsky2026/supp.zip "$EPMC/PMC13216088/supplementaryFiles"   # Yakubovsky 2026 Nature
unzip -oq halpern2017/supp.zip '*.xlsx' '*.docx' -d halpern2017/
unzip -oq brosch2018/supp.zip '*.xlsx' '*.docx' -d brosch2018/
(cd yakubovsky2026 && unzip -oq supp.zip '*.xlsx' 41586_2026_10377_MOESM1_ESM.zip \
   && unzip -oq 41586_2026_10377_MOESM1_ESM.zip '*table_2.xlsx' '*table_7.xlsx' '*Guide.docx')

MART=https://www.ensembl.org/biomart/martservice
q() { echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?><!DOCTYPE Query><Query virtualSchemaName=\"default\" formatter=\"TSV\" header=\"1\" uniqueRows=\"1\" datasetConfigVersion=\"0.6\"><Dataset name=\"btaurus_gene_ensembl\" interface=\"default\">$1</Dataset></Query>"; }
curl -sL -G --data-urlencode "query=$(q '<Attribute name="ensembl_gene_id"/><Attribute name="external_gene_name"/><Attribute name="hsapiens_homolog_associated_gene_name"/><Attribute name="hsapiens_homolog_orthology_type"/><Attribute name="mmusculus_homolog_associated_gene_name"/><Attribute name="mmusculus_homolog_orthology_type"/>')" \
  "$MART" -o orthologs/cow_human_mouse_orthologs.tsv
curl -sL -G --data-urlencode "query=$(q '<Attribute name="ensembl_gene_id"/><Attribute name="external_gene_name"/><Attribute name="entrezgene_accession"/><Attribute name="entrezgene_id"/>')" \
  "$MART" -o orthologs/cow_ensembl_ncbi_xref.tsv

"$PY" -I - <<'EOF'
import csv, openpyxl, pandas as pd
def dump(f, sheet, out, skip):
    wb = openpyxl.load_workbook(f, read_only=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        for i, r in enumerate(wb[sheet].iter_rows(values_only=True)):
            if i >= skip:
                w.writerow(r)
    wb.close()
y = "yakubovsky2026/2025-01-01424E-s1/"
dump(y + "supplementary_table_2.xlsx", "mean zonation of non steatotic", "csv/human_LHD5_zonation.csv", 1)
for s in openpyxl.load_workbook(y + "supplementary_table_2.xlsx", read_only=True).sheetnames[1:]:
    dump(y + "supplementary_table_2.xlsx", s, f"csv/human_sample_{s}.csv", 1 if s == "M2" else 0)  # per donor (04)
names = openpyxl.load_workbook(y + "supplementary_table_7.xlsx", read_only=True).sheetnames
for s in names:
    dump(y + "supplementary_table_7.xlsx", s, f"csv/nonhuman_{s}.csv", 1 if s == names[0] else 0)
dump("halpern2017/NIHMS70855-supplement-Supplementary_Table_3.xlsx", "Zonation matrix", "csv/mouse_halpern2017_S3.csv", 2)
pd.read_excel("brosch2018/41467_2018_6611_MOESM2_ESM.xlsx", header=[0, 1]).to_csv("csv/human_brosch2018_LCM.csv", index=False)
pd.read_excel("yakubovsky2026/41586_2026_10377_MOESM5_ESM.xlsx", sheet_name="Fig3a").to_csv(
    "csv/yakubovsky_fig3a_mouse_human.csv", index=False)
EOF
echo "references ready in $R"
