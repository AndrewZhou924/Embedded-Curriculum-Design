from PIL import Image, ImageFilter
 
def processing(img, filterList):
    for f in filterList:
        if   f == 'MedianFilter':
            Image_filter = ImageFilter.MedianFilter
        elif f == 'GaussianBlur':
            Image_filter = ImageFilter.GaussianBlur
        elif f == 'BLUR':
            Image_filter = ImageFilter.BLUR
        elif f == 'SHARPEN':
            Image_filter = ImageFilter.SHARPEN
        elif f == 'DETAIL':
            Image_filter = ImageFilter.DETAIL
        elif f == 'EDGE_ENHANCE':
            Image_filter = ImageFilter.EDGE_ENHANCE 
        elif f == 'ModeFilter':
            Image_filter = ImageFilter.ModeFilter(size=5)
            
            
        img = img.filter(Image_filter)
    
    return img
    
im  = Image.open('./app/image/en_words/interesting.jpg')
# im2 = im.filter(ImageFilter.BLUR)     # 模糊滤波
# im2 = im.filter(ImageFilter.GaussianBlur(radius=2)) # 高斯模糊滤波
# im2 = im.filter(ImageFilter.MedianFilter) 
# im2 = im2.filter(ImageFilter.MedianFilter) 
# im2 = im2.filter(ImageFilter.MedianFilter) 
# im3 = im2.filter(ImageFilter.SHARPEN) # 锐化滤波
# im4 = im2.filter(ImageFilter.DETAIL) # 细节滤波

# im.show()
# im2.show()
# im3.show()
# im4.show()
# im3.save('./app/image/en_words/java_clean.jpg')

list1 = [
    'BLUR',
    
    'MedianFilter',
    
    'SHARPEN',
    
    'ModeFilter',
    
    'MedianFilter',
    
    'SHARPEN',
]

clean_im = processing(im, list1)
# clean_im.show()
clean_im.save('./app/image/en_words/interesting_clean2.jpg')