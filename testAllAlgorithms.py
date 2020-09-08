from cnnRecognition.app.CNNinference import chineseRecognizeSingleImage, recognizeEngAndNumber
from pix2pix.GAN_inference_warpper   import GAN_generate
from quickDraw.quickDrawInference    import QDinference, getQDmodel
from PIL import Image

if __name__ == "__main__":
    # test character recognition API OK
    imgPath    = './cnnRecognition/app/image/all_single_cnn/peng.jpg'
    CNNresult  = chineseRecognizeSingleImage(imgPath)
    ENGresult  = recognizeEngAndNumber(imgPath)
    print("==> CNNresult:\n", CNNresult['pred1_cnn'], CNNresult['pred1_accuracy'])
    print("==> ENGresult:\n", ENGresult)
    
    # test quickDraw
    imgPath = './quickDraw/testData/new/cat.jpg'
    net     = getQDmodel()
    [pred, pred_cls] = QDinference(imgPath, net=net)
    print(pred, pred_cls)
    
    # test GAN OK
    imgPath    = './pix2pix/images/test.png'
    resultPath = GAN_generate(imgPath)
    resultImg  = Image.open(resultPath)
    resultImg.show()