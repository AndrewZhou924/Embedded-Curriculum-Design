from cnnRecognition.app.CNNinference import chineseRecognizeSingleImage, recognizeEngAndNumber

if __name__ == "__main__":
    # test character recognition API
    imgPath    = './cnnRecognition/app/image/all_single_cnn/peng.jpg'
    CNNresult  = chineseRecognizeSingleImage(imgPath)
    ENGresult  = recognizeEngAndNumber(imgPath)
    print("==> CNNresult:\n", CNNresult['pred1_cnn'], CNNresult['pred1_accuracy'])
    print("==> ENGresult:\n", ENGresult)
    
    # test quickDraw
    
    # test GAN