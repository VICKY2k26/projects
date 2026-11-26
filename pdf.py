from fpdf import FPDF


class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=9)
        self.cell(0, 10, text=f"Page {self.page_no()}", align="C")


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

pdf.set_font("Helvetica", style="B", size=18)
pdf.cell(0, 12, text="Python PDF Example", new_x="LMARGIN", new_y="NEXT")

pdf.ln(4)
pdf.set_font("Helvetica", size=12)
text = (
    "This PDF file was generated using Python and fpdf2. "
    "multi_cell wraps long paragraphs automatically inside the page margins "
    "and adds new pages when needed."
)
pdf.multi_cell(0, 8, text=text)

pdf.output("output.pdf")