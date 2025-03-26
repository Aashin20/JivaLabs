import cv2 as cv
import imutils
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import numpy as np
import os
from reportlab.lib import colors

def process_and_save_images(input_image_path):
    image = cv.imread(input_image_path)
    image = cv.resize(image, (500, 500))
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite('gray.png', gray)
    
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    cv.imwrite('blurred.png', blurred)
    
    thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
    cv.imwrite('threshold.png', thresh)
    
    eroded = cv.erode(thresh, None, iterations=2)
    cv.imwrite('eroded.png', eroded)
    
    dilated = cv.dilate(thresh, None, iterations=2)
    cv.imwrite('dilated.png', dilated)
    
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    contour_image = image.copy()
    cv.drawContours(contour_image, cnts, -1, (0, 255, 0), 2)
    cv.imwrite('contours.png', contour_image)

def create_pdf(final_result):  
    c = canvas.Canvas("image_processing_steps.pdf", pagesize=letter)
    width, height = letter
    

    logo_path = 'logo.jpeg'  
    logo_width = 2*inch 
    logo_height = 1*inch
    
    logo_x = (width - logo_width) / 2
    logo_y = height - 1.5*inch
 
    c.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height, preserveAspectRatio=True)
    c.drawString(logo_x- 0.5*inch,logo_y,"Accessible Healthcare At Your Fingertips")
  
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(0.5*inch, logo_y - 0.25*inch, width - 1*inch, logo_y - 0.25*inch)
   
    c.setFont("Helvetica", 12)
    


    left_x = 1*inch
    info_y = logo_y - 1*inch  
    c.setFont("Helvetica", 12)
    c.drawString(left_x, info_y, "Test Type: ")
    c.setFont("Helvetica", 12)
    c.drawString(left_x + c.stringWidth("Test Type: ", "Helvetica", 12), info_y, "Brain Lesion Diagnostic Imaging")

    c.setFont("Helvetica", 12)
    c.drawString(left_x, info_y - 0.3*inch, "Patient Name: ")
    c.setFont("Helvetica", 12)
    c.drawString(left_x + c.stringWidth("Patient Name: ", "Helvetica", 12), info_y - 0.3*inch, "Jeswin Sunsi")

    c.setFont("Helvetica", 12)
    c.drawString(left_x, info_y - 0.6*inch, "Age: ")
    c.setFont("Helvetica", 12)
    c.drawString(left_x + c.stringWidth("Age: ", "Helvetica", 12), info_y - 0.6*inch, "19")
        
  
    right_x = width/2 + 0.7*inch
    c.setFont("Helvetica", 12)
    c.drawString(right_x, info_y, "Sex:")
    c.drawString(right_x, info_y - 0.3*inch, "Date:")
    

    c.setFont("Helvetica", 12)
    
    
   
    c.drawString(right_x + 0.5*inch, info_y, "Male")
    c.drawString(right_x + 0.5*inch, info_y - 0.3*inch, "23-03-25")
    
    
    c.setFont("Helvetica", 12)
    prob_y = info_y - 1.2*inch  
    c.drawString(left_x, info_y-0.9*inch, "Provisional Result:")
    c.setFont("Helvetica", 12)
    result=""
    result="Yes" if final_result>=60 else "No"
    c.drawString(left_x + 1.5*inch, info_y-0.9*inch, f"{result}")
    

    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(0.5*inch, prob_y - 0.5*inch, width - 1*inch, prob_y - 0.5*inch)
    
    
    img_width = width/3.5 - 0.50*inch
    img_height = height/3.5 - 0.50*inch
    
    horizontal_spacing = 0.7*inch  
    vertical_spacing = 0.1*inch   
    
    images = [
        ('gray.png', 'NeuroSpectral View'),
        ('blurred.png', 'Diffusion Map'),
        ('threshold.png', 'Tumor Segmentation'),
        ('eroded.png', 'Morphological Reduction'),
        ('dilated.png', 'Lesion Enhancement'),
        ('contours.png', 'Neoplastic Boundary Detection')
    ]
    
    
    for i, (img_path, title) in enumerate(images):
        row = i // 3
        col = i % 3
        
        x = horizontal_spacing + col * (img_width + horizontal_spacing)
        
        y = height - 4.5*inch - (vertical_spacing + img_height) - row * (img_height + vertical_spacing)
        
        c.setFont("Helvetica", 12)
        c.drawString(x, y + img_height + 0.1*inch, title)
        c.drawImage(img_path, x, y, width=img_width, height=img_height, preserveAspectRatio=True)
    
    c.save()
    for img_path, _ in images:
        if os.path.exists(img_path):
            os.remove(img_path)
def main():
    input_image_path = 'Y104.jpg'
    final_result = 95.6
    process_and_save_images(input_image_path)
    create_pdf(final_result) 
    print("PDF created successfully!")

if __name__ == "__main__":
    main()