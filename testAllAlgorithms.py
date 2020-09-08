from cnnRecognition.app.CNNinference import chineseRecognizeSingleImage, recognizeEngAndNumber

if __name__ == "__main__":
    # test character recognition API
    imgPath    = './cnnRecognition/app/image/test.png'
    CNNresult  = chineseRecognizeSingleImage(imgPath)
    ENGresult  = recognizeEngAndNumber(imgPath)
    print("==> CNNresult:\n", CNNresult)
    print("==> ENGresult:\n", ENGresult)
    
    # test quickDraw
    
    # test GAN