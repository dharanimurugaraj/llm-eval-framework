from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

os.makedirs('data/raw', exist_ok=True)
c = canvas.Canvas('data/raw/financial_report_2024.pdf', pagesize=letter)

pages = [
    [
        'ACME Corporation — Q3 2024 Financial Report',
        'Executive Summary',
        'Revenue increased 23% year-over-year to 4.2 billion dollars.',
        'Operating margin expanded 150 basis points to 18.3 percent.',
        'Free cash flow generation remained robust at 890 million dollars.',
        'The board approved a new share buyback program of 500 million dollars.',
        'Headcount grew to 42000 employees globally across all divisions.',
    ],
    [
        'Revenue Breakdown by Segment',
        'Cloud Services revenue grew 45% to 1.8 billion dollars.',
        'Enterprise Software revenue grew 12% to 1.4 billion dollars.',
        'Professional Services revenue grew 8% to 600 million dollars.',
        'Hardware division revenue declined 5% to 400 million dollars.',
        'International revenue now represents 38% of total revenue.',
        'North America remains the largest market at 62% of revenue.',
    ],
    [
        'Risk Factors and Outlook',
        'Management raised full-year guidance to 16.5 billion dollars.',
        'Key risks include macroeconomic uncertainty and FX headwinds.',
        'Competition in cloud services intensified during the quarter.',
        'Supply chain normalization contributed to margin improvement.',
        'R&D investment increased 18% to support next generation products.',
        'Customer retention rate improved to 94% from 91% last year.',
    ],
    [
        'Balance Sheet Highlights',
        'Total assets stood at 28.4 billion dollars at quarter end.',
        'Cash and equivalents increased to 6.2 billion dollars.',
        'Long-term debt reduced by 800 million dollars during the quarter.',
        'Debt to equity ratio improved to 0.42 from 0.51 a year ago.',
        'Working capital position remains strong at 4.1 billion dollars.',
        'Capital expenditure was 620 million dollars for the quarter.',
    ],
]

for i, page_lines in enumerate(pages):
    y = 750
    for j, line in enumerate(page_lines):
        font = 'Helvetica-Bold' if j == 0 else 'Helvetica'
        size = 14 if j == 0 else 11
        c.setFont(font, size)
        c.drawString(72, y, line)
        y -= 30
    c.showPage()

c.save()
print('Created: data/raw/financial_report_2024.pdf (4 pages)')
