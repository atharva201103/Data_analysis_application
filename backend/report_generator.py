"""
Report Generation Engine for DataSense AI.
Produces downloadable PDF reports via ReportLab, Excel workbooks via openpyxl,
and clean CSV data exports.
"""

import io
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak, HRFlowable
)


class ReportGenerator:
    """Generates PDF, Excel, and CSV analytical deliverables."""

    @staticmethod
    def generate_pdf_report(
        df: pd.DataFrame,
        profile_data: Dict[str, Any],
        kpis: List[Dict[str, Any]],
        insights: List[Dict[str, Any]],
        cleaning_log: List[Dict[str, Any]],
        dataset_name: str = "Uploaded Dataset"
    ) -> bytes:
        """
        Build an executive PDF analytical report using ReportLab.
        Returns PDF bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#4F46E5"),
            spaceAfter=15
        )
        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=14,
            spaceAfter=8
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#334155")
        )
        insight_title = ParagraphStyle(
            "InsightTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#1E293B")
        )
        rec_style = ParagraphStyle(
            "RecText",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#059669")
        )

        story = []

        # 1. Header Banner
        story.append(Paragraph("DataSense AI", title_style))
        story.append(Paragraph(f"Automated Executive Analytical Report • {dataset_name}", subtitle_style))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')} | Engine Version 1.0.0", body_style))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#E2E8F0"), spaceAfter=15))

        # 2. Executive Summary & Health Score
        story.append(Paragraph("1. Dataset Overview & Data Quality Health", section_style))
        health_score = profile_data.get("health_score", 100)
        grade = profile_data.get("health_grade", "A")

        score_color = "#10B981" if health_score >= 80 else ("#F59E0B" if health_score >= 60 else "#EF4444")

        overview_text = (
            f"The analyzed dataset contains <b>{len(df):,} records</b> across <b>{len(df.columns)} attributes</b> "
            f"({profile_data.get('memory_formatted', 'N/A')} memory footprint). "
            f"Total missing cells: <b>{profile_data.get('total_missing_cells', 0):,}</b> "
            f"({profile_data.get('total_missing_percentage', 0):.1f}%). "
            f"Exact duplicate rows detected: <b>{profile_data.get('duplicate_rows', 0):,}</b>."
        )
        story.append(Paragraph(overview_text, body_style))
        story.append(Spacer(1, 8))

        # Summary Metrics Table
        metric_table_data = [
            ["Health Score", f"{health_score}/100 ({grade})", "Total Rows", f"{len(df):,}"],
            ["Numerical Features", str(len(profile_data.get("numeric_columns", []))), "Total Columns", str(len(df.columns))],
            ["Categorical Features", str(len(profile_data.get("categorical_columns", []))), "Duplicates", f"{profile_data.get('duplicate_rows', 0):,}"],
            ["Missing Cell Ratio", f"{profile_data.get('total_missing_percentage', 0):.1f}%", "Memory Footprint", profile_data.get("memory_formatted", "N/A")]
        ]
        t_metric = Table(metric_table_data, colWidths=[130, 130, 130, 140])
        t_metric.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1E293B")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_metric)
        story.append(Spacer(1, 15))

        # 3. Key Business KPIs
        if kpis:
            story.append(Paragraph("2. Executive Business KPIs", section_style))
            kpi_rows = [["KPI Indicator", "Value", "Operational Domain", "Context / Subtext"]]
            for k in kpis:
                kpi_rows.append([
                    k.get("title", ""),
                    str(k.get("value", "")),
                    k.get("category", "General"),
                    k.get("subtext", "")
                ])
            t_kpi = Table(kpi_rows, colWidths=[140, 110, 110, 170])
            t_kpi.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t_kpi)
            story.append(Spacer(1, 15))

        # 4. Key Business Findings & Recommendations
        if insights:
            story.append(Paragraph("3. Strategic Findings & Recommendations", section_style))
            for idx, ins in enumerate(insights[:5], 1):
                p_title = Paragraph(f"<b>{idx}. {ins.get('title', 'Insight')}</b>", insight_title)
                p_obs = Paragraph(f"<b>Observation:</b> {ins.get('observation', '')}", body_style)
                p_imp = Paragraph(f"<b>Business Implication:</b> {ins.get('implication', '')}", body_style)
                p_rec = Paragraph(f"<b>Strategic Recommendation:</b> {ins.get('recommendation', '')}", rec_style)

                table_data = [[p_title], [p_obs], [p_imp], [p_rec]]
                t_ins = Table(table_data, colWidths=[530])
                t_ins.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
                    ("LINELEFT", (0, 0), (0, -1), 3.5, colors.HexColor("#4F46E5")),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(t_ins)
                story.append(Spacer(1, 8))

        # 5. Cleaning Audit Trail
        if cleaning_log:
            story.append(Spacer(1, 8))
            story.append(Paragraph("4. Data Cleaning Audit Trail", section_style))
            clean_rows = [["Timestamp", "Transformation", "Rows Affected", "Audit Details"]]
            for op in cleaning_log[-6:]:
                clean_rows.append([
                    op.get("timestamp", "")[:19],
                    op.get("operation", ""),
                    str(op.get("rows_affected", 0)),
                    op.get("details", "")[:80]
                ])
            t_clean = Table(clean_rows, colWidths=[95, 120, 75, 240])
            t_clean.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t_clean)

        # 6. Assumptions & Limitations
        story.append(Spacer(1, 14))
        story.append(Paragraph("5. Limitations and Analytical Assumptions", section_style))
        disclaimer_text = (
            "<b>Disclaimer:</b> All figures and patterns in this report were deterministically calculated from "
            "the provided dataset. DataSense AI uses rigorous statistical methods; however, correlations observed "
            "do not necessarily imply direct causation. Business recommendations should be validated against internal "
            "domain context and market conditions prior to operational deployment."
        )
        story.append(Paragraph(disclaimer_text, body_style))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def generate_excel_workbook(
        cleaned_df: pd.DataFrame,
        profile_data: Dict[str, Any],
        kpis: List[Dict[str, Any]],
        cleaning_log: List[Dict[str, Any]],
        eda_summary: Optional[pd.DataFrame] = None
    ) -> bytes:
        """
        Build a multi-tab professional Excel report with styled sheets.
        Returns Excel workbook bytes.
        """
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Sheet 1: Cleaned Data
            cleaned_df.to_excel(writer, sheet_name="Cleaned_Data", index=False)

            # Sheet 2: Executive KPIs
            if kpis:
                kpi_df = pd.DataFrame([
                    {
                        "Indicator": k.get("title"),
                        "Value": k.get("value"),
                        "Domain": k.get("category"),
                        "Context": k.get("subtext")
                    } for k in kpis
                ])
                kpi_df.to_excel(writer, sheet_name="Executive_KPIs", index=False)

            # Sheet 3: Data Quality Profile
            col_details = profile_data.get("column_details", [])
            if col_details:
                profile_df = pd.DataFrame(col_details)
                profile_df.to_excel(writer, sheet_name="Column_Quality_Profile", index=False)

            # Sheet 4: EDA Descriptive Statistics
            if eda_summary is not None and not eda_summary.empty:
                eda_summary.to_excel(writer, sheet_name="Descriptive_Statistics", index=False)

            # Sheet 5: Cleaning Audit Log
            if cleaning_log:
                clean_df = pd.DataFrame(cleaning_log)
                clean_df.to_excel(writer, sheet_name="Cleaning_Audit_Log", index=False)

        buffer.seek(0)
        return buffer.getvalue()

