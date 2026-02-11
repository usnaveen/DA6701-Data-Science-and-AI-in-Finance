# Portfolio Report LaTeX Template

Template file:
- `/Users/tanmaygawande/Documents/New project/Portfolio_Report_LaTeX_Template.tex`

## What this matches
- A4 layout with tight margins (`1.6cm` left/right, `1.3cm` top/bottom)
- Helvetica-style typography
- Dark blue title/section bars
- Two KPI blocks (GMV + MSR) with colored metric emphasis
- Side-by-side figure panel section
- Full-width styled weights table with alternating row fill
- Thin footer separator and metadata line

## What to edit for each assignment
Inside the `.tex` file, update these variables:
- `\ReportTitle`
- `\ReportSubtitle`
- `\GMVReturn`, `\GMVVol`, `\GMVSharpe`
- `\MSRReturn`, `\MSRVol`, `\MSRSharpe`
- `\MethodologyText`
- `\EffFrontierFig`, `\CorrFig`
- `\FooterText`

To update the table values, replace rows in **Section 4**.

## Compile
Use one of:
- `pdflatex Portfolio_Report_LaTeX_Template.tex`
- `xelatex Portfolio_Report_LaTeX_Template.tex`
- `lualatex Portfolio_Report_LaTeX_Template.tex`

Note: LaTeX compilers are not currently installed in this environment, so compilation was not run here.
