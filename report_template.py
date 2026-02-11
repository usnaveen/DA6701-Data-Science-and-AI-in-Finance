import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

class PortfolioReport:
    def __init__(self, output_path, title, subtitle):
        self.W, self.H = A4
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=1.6*cm, rightMargin=1.6*cm,
            topMargin=1.4*cm,  bottomMargin=1.4*cm
        )
        
        # Color Palette
        self.BLUE = colors.HexColor("#1a3c6e")
        self.LGRAY = colors.HexColor("#f5f5f5")
        self.MGRAY = colors.HexColor("#d0d0d0")
        self.GOLD = colors.HexColor("#e8a020")
        self.GREEN = colors.HexColor("#2d7a4f")
        
        # Styles
        self.styles = getSampleStyleSheet()
        self._init_styles()
        
        # Story (Content)
        self.story = []
        self._add_title_section(title, subtitle)

    def _init_styles(self):
        self.title_style = ParagraphStyle("title", fontSize=14, leading=17,
            textColor=self.BLUE, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=2)
        
        self.sub_style = ParagraphStyle("sub", fontSize=8.5, leading=11,
            textColor=colors.HexColor("#555555"), alignment=TA_CENTER, fontName="Helvetica", spaceAfter=4)
        
        self.sec_style = ParagraphStyle("sec", fontSize=9.5, leading=12,
            textColor=colors.white, fontName="Helvetica-Bold", spaceAfter=0)
        
        self.body_style = ParagraphStyle("body", fontSize=7.8, leading=10.5,
            textColor=colors.HexColor("#333333"), fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=3)
        
        self.label_style = ParagraphStyle("lbl", fontSize=7.5, leading=9,
            textColor=colors.HexColor("#666666"), fontName="Helvetica", alignment=TA_CENTER)

    def _add_title_section(self, title, subtitle):
        self.story.append(Paragraph(title, self.title_style))
        self.story.append(Paragraph(subtitle, self.sub_style))
        self.story.append(HRFlowable(width="100%", thickness=1.5, color=self.BLUE, spaceAfter=5))

    def add_section_header(self, text):
        tbl = Table([[Paragraph(text, self.sec_style)]], colWidths=[self.W - 3.2*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), self.BLUE),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
        ]))
        self.story.append(Spacer(1, 5))
        self.story.append(tbl)
        self.story.append(Spacer(1, 4))

    def add_paragraph(self, text):
        self.story.append(Paragraph(text, self.body_style))

    def add_image_row(self, img_paths, captions, widths):
        """Adds 1 or 2 images side by side with labels."""
        imgs = [Image(p, width=w, height=6.5*cm) for p, w in zip(img_paths, widths)]
        tbl = Table([imgs], colWidths=widths)
        tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("ALIGN", (0,0), (-1,-1), "CENTER")]))
        self.story.append(tbl)
        
        caps = [Paragraph(c, self.label_style) for c in captions]
        cap_tbl = Table([caps], colWidths=widths)
        self.story.append(cap_tbl)

    def add_data_table(self, data, col_widths, is_kpi=False):
        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        
        # Shared logic for table styling
        base_style = [
            ("BACKGROUND", (0,0), (-1,0), self.BLUE),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 7.5 if not is_kpi else 8),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.4, self.MGRAY),
            ("BOX", (0,0), (-1,-1), 0.8, self.BLUE),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [self.LGRAY, colors.white]),
        ]
        
        if is_kpi:
            base_style.extend([
                ("TEXTCOLOR", (2, 1), (2, -1), self.GREEN),
                ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold")
            ])
            
        tbl.setStyle(TableStyle(base_style))
        self.story.append(Spacer(1, 5))
        self.story.append(tbl)

    def add_footer(self, text):
        self.story.append(HRFlowable(width="100%", thickness=0.8, color=self.MGRAY, spaceBefore=10))
        footer_style = ParagraphStyle("footer", fontSize=6.5, textColor=colors.HexColor("#999999"),
                                    alignment=TA_CENTER, fontName="Helvetica")
        self.story.append(Paragraph(text, footer_style))

    def save(self):
        self.doc.build(self.story)
