import pandas as pd
import numpy as np
class Nueral_Network:
    def __init__(self,dataset,y_test,nueron=[3,1],opti='momentum',weight_init='He',type='regression',learning_rate=0.01,epoch=1000):
        self.rate = learning_rate
        self.count=0
        self.nueron=nueron
        self.class_index = {}
        self.der_z={}
        self.der_b={}
        self.der_w={}
        self.velocity={}
        self.m_velocity={}
        self.bais_velocity={}
        self.m_bais = {}
        self.v_bais={}
        
        if(opti=='momentum'):
            self.update_parameter = self.momentum_update
        elif(opti=='Adam'):
            self.update_parameter=self.adam_update
        self.dataset = dataset
        
        if(nueron[-1]>1):
            
            self.target=self.one_hot_coding(y_test)
            
        else:
            self.target = y_test.values.reshape(-1,1)
            
        self.layer=len(nueron)
        self.node = nueron
        
        self.bais_velocity['V1']=np.zeros((1,self.node[0]))
        self.velocity['V1']=np.zeros((self.node[0],self.dataset.shape[1]))
        self.m_velocity['m1']=np.zeros((self.node[0],self.dataset.shape[1]))

        self.m_bais['mb1']=np.zeros((1,self.node[0]))
        self.v_bais['vb1']=np.zeros((1,self.node[0]))


        
        for i in range(2,self.layer+1):
            self.velocity['V'+str(i)]=np.zeros((self.node[i-1],self.node[i-2]))
            self.m_velocity['m'+str(i)]=np.zeros((self.node[i-1],self.node[i-2]))
            
            self.bais_velocity['V'+str(i)]=np.zeros((1,self.node[i-1]))
            self.m_bais['mb'+str(i)]=np.zeros((1,self.node[i-1]))
            self.v_bais['vb'+str(i)]=np.zeros((1,self.node[i-1]))
            
            

        
        if(nueron[-1]>1):
            self.activation=self.softmax
            self.loss=self.softmax_loss
            self.last_der = self.softmax_last_der
            
        elif(nueron[-1]==1 and type=='regression'):
            self.activation=self.regression
            self.loss=self.regression_loss
            self.last_der = self.regression_last_der

        elif(nueron[-1]==1 and type=='binary'):
            self.activation=self.sigmoid
            self.loss=self.sigmoid_loss
            self.last_der = self.sigmoid_last_der
            
        if(weight_init=='Xavier'):
            self.weights,self.bais = self.Xavier()
            
        elif(weight_init=='He'):
            self.weights,self.bais = self.He()
            
        for i in range(epoch):
            self.z,self.a = self.forward()
            y_hat = self.a['A'+str(self.layer)]
            loss_cal = self.loss(y_hat)
            #===========================backpropagation==============
            self.der_z['dz'+str(self.layer)]=self.last_der(y_hat)
            self.der_w['dw'+str(self.layer)]=(1/self.dataset.shape[0])*(np.dot((self.der_z['dz'+str(self.layer)]).T,(self.a['A'+str(self.layer-1)])))
            self.der_b['db'+str(self.layer)] = (1/self.dataset.shape[0])*np.sum(self.der_z['dz'+str(self.layer)],axis=0,keepdims=True)
           #======================last_layer_ko_chhod_kar================
            for k in reversed(range(1,self.layer)):
                self.der_z['dz'+str(k)]=(np.dot((self.der_z['dz'+str(k+1)]),(self.weights['W'+str(k+1)])))*self.der_relu(self.z['Z'+str(k)])
                self.der_w['dw'+str(k)]=(1/self.dataset.shape[0])*np.dot(self.der_z['dz'+str(k)].T,self.a['A'+str(k-1)])
                self.der_b['db'+str(k)] = (1/self.dataset.shape[0])*np.sum(self.der_z['dz'+str(k)],axis=0,keepdims=True)
            #======================Updating_parameter==================
            self.update_parameter()
            if(i%100==0):
                print(f'Loss after {i} iteration is {loss_cal}')

        print(f"Final loss is {loss_cal}")
        print('''
        (\___/)
        (^ _ ^)
        (>Training complete💕
        ''')
        
        

    def one_hot_coding(self,y_test):
        y_test = np.array(y_test)
        
        pos = 0
        for i in y_test:
            if i not in self.class_index:
                self.class_index[i]=pos
                pos = pos+1
        target = []
        for i in y_test:
            new = []
            for j in range(self.nueron[-1]):
                if(j==self.class_index[i]):
                    new.append(1)
                else:
                    new.append(0)
            target.append(new)
        return np.array(target)
        
    def adam_update(self):
        beta1=0.9
        beta2=0.99
        eps=1e-8
        self.count = self.count+1
        
        for i in range(1,self.layer+1):
            self.m_velocity['m'+str(i)]=beta1*self.m_velocity['m'+str(i)]+(1-beta1)*self.der_w['dw'+str(i)]
            self.velocity['V'+str(i)]=beta2*self.velocity['V'+str(i)]+(1-beta2)*((self.der_w['dw'+str(i)])**2)
            self.weights['W'+str(i)]=self.weights['W'+str(i)]-((self.rate)/((self.velocity['V'+str(i)]+eps)**0.5))*self.m_velocity['m'+str(i)]

            self.m_bais['mb'+str(i)]= beta1*self.m_bais['mb'+str(i)]+(1-beta1)*(self.der_b['db'+str(i)])
            self.v_bais['vb'+str(i)]= beta2*self.v_bais['vb'+str(i)]+(1-beta2)*((self.der_b['db'+str(i)])**2)

            mb_hat = self.m_bais['mb'+str(i)] / (1 - beta1 ** self.count)
            vb_hat = self.v_bais['vb'+str(i)] / (1 - beta2 ** self.count)
            self.bais['b'+str(i)] = self.bais['b'+str(i)]-(self.rate * mb_hat / (np.sqrt(vb_hat) + eps))
            
            



            
            
        
        
    def momentum_update(self):
        for i in range(1,self.layer+1):
            
            self.velocity['V'+str(i)]=0.9*self.velocity['V'+str(i)]+(self.rate)*self.der_w['dw'+str(i)]
            self.weights['W'+str(i)] = self.weights['W'+str(i)]-self.velocity['V'+str(i)]

            self.bais_velocity['V'+str(i)]=0.9*self.bais_velocity['V'+str(i)]+(self.rate)*self.der_b['db'+str(i)]
            self.bais['b'+str(i)] = self.bais['b'+str(i)]-self.bais_velocity['V'+str(i)]
            
        
    def predict(self,test_dataset):
        test_dataset = np.array(test_dataset)
        if (test_dataset.ndim==1):
            test_dataset = test_dataset.reshape(1,-1)
        input = test_dataset
        for i in range(1,self.layer):
            z=np.dot(input,self.weights['W'+str(i)].T)+self.bais['b'+str(i)]
            a = self.relu(z)
            input =a
        z = np.dot(input,self.weights['W'+str(self.layer)].T)+self.bais['b'+str(self.layer)]
        output = self.activation(z)
        output = np.array(output)
        
        if(self.activation==self.sigmoid):
            #print("I am inside activation")
            result =[]
            
            for i in range(len(output)):
                if(output[i]>0.5):
                    result.append(1)
                else:
                    result.append(0)
            return result
            
        if(self.activation==self.softmax):
            
            result= np.argmax(output,axis=1)
            final_result=[]
            reverse_dic = {}
            
            for key,value in self.class_index.items():
                reverse_dic[value]=key
            
            for i in result:
                final_result.append(reverse_dic[i])
                
            return final_result
            
        return output
        
            
    def forward(self): 
        z={}
        a={}
        a['A0']=self.dataset
        for i in range(1,self.layer):
            z['Z'+str(i)]=np.dot(a['A'+str(i-1)],self.weights['W'+str(i)].T)+self.bais['b'+str(i)]
            a['A'+str(i)]=self.relu(z['Z'+str(i)])
        z['Z'+str(self.layer)]=np.dot(a['A'+str(self.layer-1)],self.weights['W'+str(self.layer)].T)+self.bais['b'+str(self.layer)]
        a['A'+str(self.layer)]=self.activation(z['Z'+str(self.layer)])
        return z,a

    def sigmoid(self,value):
        value = np.clip(value,-500,500)
        return 1/(1+np.exp(-value))

    def sigmoid_loss(self,y_hat):
        eps = 1e-8
        loss = -(self.target*(np.log(y_hat+eps))+(1-self.target)*(np.log(1-y_hat+eps)))
        return np.mean(loss)
    
    def sigmoid_last_der(self,y_hat):
        return y_hat-self.target
        
    def regression(self,value):
        return value
        
    def regression_loss(self,y_hat):
        #y_hat = np.clip(y_hat, -1e6, 1e6)
        loss = (y_hat-self.target)**2
        return np.mean(loss)
        
    def regression_last_der(self,y_hat):
        
        last=-2*(self.target-y_hat)
        return last
        

        
    def softmax(self,z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        
    def softmax_loss(self,y_hat):
        eps = 1e-8
        y_hat = np.clip(y_hat, eps, 1 - eps)
        loss = -(np.sum(self.target*(np.log(y_hat)),axis=1))
        return np.mean(loss)
        
    def softmax_last_der(self,y_hat):
        return y_hat-self.target
        
    def relu(self,z):
        return np.maximum(0,z)
        
    def der_relu(self,z):
        return (z > 0).astype(float)
        
    def Xavier(self):
        weights = {}
        bais = {}
        weights['W1']=np.random.randn(self.node[0],self.dataset.shape[1])*((6/(self.node[0]+self.dataset.shape[1]))**0.5)
        bais['b1']=np.zeros((1,self.node[0]))
        for i in range(2,self.layer+1):
            weights['W'+str(i)]=np.random.randn(self.node[i-1],self.node[i-2])*((6/(self.node[i-1]+self.node[i-2]))**0.5)
            bais['b'+str(i)]=np.zeros((1,self.node[i-1]))
        return weights,bais
    def He(self):
        weights = {}
        bais = {}
        weights['W1']=np.random.randn(self.node[0],self.dataset.shape[1])*((2/(self.dataset.shape[1]))**0.5)
        bais['b1']=np.zeros((1,self.node[0]))
        for i in range(2,self.layer+1):
            weights['W'+str(i)]=np.random.randn(self.node[i-1],self.node[i-2])*((2/(self.node[i-2]))**0.5)
            bais['b'+str(i)]=np.zeros((1,self.node[i-1]))
        return weights,bais

        

import cv2
from scipy.ndimage import center_of_mass
def convert_image(path):
# 1. Read image
    img = cv2.imread(path)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Gaussian blur (remove camera noise)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Adaptive threshold (FILLED strokes, not edges)
    gray = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # 5. Thicken strokes (CRITICAL for ANN)
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    
    # 6. Crop tightly around digit
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    digit = gray[y:y+h, x:x+w]
    
    # 7. Resize to MNIST core (20×20)
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    
    # 8. Pad to 28×28 (MNIST style)
    digit = np.pad(digit, ((4, 4), (4, 4)), mode="constant")
    
    # 9. Center digit using center of mass
    cy, cx = center_of_mass(digit)
    h, w = digit.shape
    
    shiftx = int(w/2 - cx)
    shifty = int(h/2 - cy)
    
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    digit = cv2.warpAffine(digit, M, (w, h))
    
    # 10. Normalize (EXACTLY like MNIST training)
    digit = digit / 255.0
    
    # 11. Flatten for ANN
    flatten_array = digit.reshape(1, 784)
    
    return flatten_array  # (1, 784)


