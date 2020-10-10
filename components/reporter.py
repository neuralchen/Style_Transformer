######################################################################
#  script name  : Reporter.py
#  author       : Chen Xuanhong
#  created time : 2019/9/19 19:21
#  modification time ：2019/9/23 23:37
#  modified by  : Chen Xuanhong
######################################################################

import datetime
import os

class Reporter:
    def __init__(self,reportPath):
        self.path           = reportPath
        self.withTimeStamp  = False
        self.index          = 1
        self.timeStrFormat  = '%Y-%m-%d %H:%M:%S'
        timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d%H%M%S')
        self.path = self.path + "-%s.log"%timeStr 
        if not os.path.exists(self.path):
            f = open(self.path,'w')
            f.close()
    
    def writeInfo(self,strLine):
        with open(self.path,'a+') as logf:
            timeStr = datetime.datetime.strftime(datetime.datetime.now(),self.timeStrFormat)
            logf.writelines("[%d]-[%s]-[info] %s\n"%(self.index,timeStr,strLine))
            self.index += 1
    
    def writeConfig(self,config):
        with open(self.path,'a+') as logf:
            for item in config.__dict__.items():
                text = "[%d]-[parameters] %s--%s\n"%(self.index,item[0],str(item[1]))
                logf.writelines(text)
                self.index +=1
    
    def writeModel(self,modelText):
        with open(self.path,'a+') as logf:
            logf.writelines("[%d]-[model] %s\n"%(self.index,modelText))
            self.index += 1
    
    def writeTrainLog(self,epoch,step,logText):
        with open(self.path,'a+') as logf:
            timeStr = datetime.datetime.strftime(datetime.datetime.now(),self.timeStrFormat)
            logf.writelines("[%d]-[%s]-[logInfo]-[%d]-[%d]\n %s\n"%(self.index,timeStr,epoch,step,logText))
            self.index += 1

    def writeTestLog(self,step,logText):
        with open(self.path,'a+') as logf:
            timeStr = datetime.datetime.strftime(datetime.datetime.now(),self.timeStrFormat)
            logf.writelines("[%d]-[%s]-[logInfo] %s\n"%(self.index,timeStr,logText))