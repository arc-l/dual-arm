from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.oxml import parse_xml

prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[5])

start_x = Inches(2.5)
start_y = Inches(2)
end_x = Inches(4)
end_y = Inches(4)
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1), Inches(1), Inches(2), Inches(2))
print(f"start_x: {start_x}, start_y: {start_y}, end_x: {end_x}, end_y: {end_y}")
connector = slide.shapes.add_connector(
    MSO_CONNECTOR.STRAIGHT,
    start_x, start_y,
    end_x, end_y
)
connector.line.color.rgb = RGBColor(0, 0, 0)
connector.line.width = Pt(2)
connector.line.end_arrowhead = True
line_elem = connector.line._get_or_add_ln()
line_elem.append(parse_xml("""
        <a:headEnd type="arrow" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>
"""))

connector = slide.shapes.add_connector(
    MSO_CONNECTOR.STRAIGHT,
    end_x, end_y,
    start_x, start_y
)
connector.line.color.rgb = RGBColor(0, 0, 0)
connector.line.width = Pt(2)
connector.line.end_arrowhead = True
line_elem = connector.line._get_or_add_ln()
line_elem.append(parse_xml("""
        <a:headEnd type="arrow" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>
"""))
prs.save("results/test_connector.pptx")
