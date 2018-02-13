# -*- coding: UTF-8 -*-
# http://ai.baidu.com/docs#/OCR-Python-SDK/07883957
# pip install git+https://github.com/Baidu-AIP/python-sdk.git@master
# windows如果没安装git,要先安装git,配置git路径
from aip import AipOcr
from pathlib import Path

class BaiduAI:
    
    # 定义常量  
    APP_ID = '10379743'
    API_KEY = 'QGGvDG2yYiVFvujo6rlX4SvD'
    SECRET_KEY = 'PcEAUvFO0z0TyiCdhwrbG97iVBdyb3Pk'

    # 初始化文字识别分类器
    aipOcr=AipOcr(APP_ID, API_KEY, SECRET_KEY)

    def get_file_content(self,filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()
        
    def set_ocr_result(self,filePath,result):
        with open(filePath+".txt", 'w') as fp:
            return fp.write(result)
        
    def ImageOCR(self,imagePath):
        # 定义参数变量
        options = {
            'detect_direction': 'true',
            'language_type': 'CHN_ENG',
        }
        # 网络图片文字文字识别接口,每天免费500次
        #result = self.aipOcr.webImage(self.get_file_content(imagePath),options)
        # 如果图片是url 调用示例如下
        # result = apiOcr.webImage('http://www.xxxxxx.com/img.jpg')
        
        #通用文字识别（高精度版）,每天免费50次
        result = self.aipOcr.basicAccurate(self.get_file_content(imagePath), options)

        
        self.set_ocr_result(imagePath,str(result))
        print(result)
        return(result)


ai = BaiduAI()
##path ="wenzi.png"
##ai.ImageOCR(path)

for i in range(42,43):
    filePath ="f:\\ocr\\"+str(i)+".jpg"
    if(Path(filePath).is_file()):
        ai.ImageOCR(filePath)
