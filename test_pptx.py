from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# Create a new presentation
prs = Presentation()

# Add a blank slide layout
blank_slide_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_slide_layout)

# Define position and size of the shape
left = Inches(2)
top = Inches(2)
width = Inches(4)
height = Inches(1.5)

# Add a rectangle shape
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)

# Customize the fill color
fill = shape.fill
fill.solid()
fill.fore_color.rgb = RGBColor(255, 0, 0)  # Red color

# Save the presentation
prs.save('custom_shape.pptx')
