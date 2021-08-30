from tkinter import *
import numpy as np
from matplotlib import pyplot as plt


root = Tk()
root.title("Single Layer Perceptron")
sizex = 1800
sizey = 1000
posx  = 100
posy  = 100
root.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))
inputs = np.array([
        ## The first column is the bias while the second and third are the actual inputs
        #bias  #x1  #x2
        [1,     0,   0],
        [1,     0,   1],
        [1,     1,   0],
        [1,     1,   1]
    ])
lr =1
epoch =10

class Perceptron:
    def __init__(self, epoch,lr,inputs,weights,desired_outputs):
        self.epoch = epoch
        self.lr = lr
        self.inputs = inputs
        self.weights = weights
        self.desired_outputs = desired_outputs

    def activation_function(self,weighted_sum):
        if weighted_sum>= 0:
            return 1
        else:
            return 0

    def get_weighted_sum(self,weights,inputs):
        w_s = weights.dot(inputs)
        activate = self.activation_function(w_s)
        return(activate)

    # def plot(self,inputs,weights=None,title="Prediction Matrix"):
        if len(inputs[0])==3: 
            fig,ax = plt.subplots()
            ax.set_title(title)
            ax.set_xlabel("x1")
            ax.set_ylabel("y1")

            if len(weights)>0:
                map_min=0.0
                map_max=1.1
                y_res=0.001
                x_res=0.001
                ys=np.arange(map_min,map_max,y_res)
                xs=np.arange(map_min,map_max,x_res)
                zs=[]
                for cur_y in np.arange(map_min,map_max,y_res):
                    for cur_x in np.arange(map_min,map_max,x_res):
                        zs.append(self.get_weighted_sum(weights,np.array([1.0,cur_x,cur_y])))
                xs,ys=np.meshgrid(xs,ys)
                zs=np.array(zs)
                zs = zs.reshape(xs.shape)
                cp=plt.contourf(xs,ys,zs,levels=[-1,-0.0001,0,1],colors=('b','r'),alpha=0.1)

            c1_data=[[],[]]
            c0_data=[[],[]]
            for i in range(len(inputs)):
                cur_i1 = inputs[i][1]
                cur_i2 = inputs[i][2]
                cur_y  = inputs[i][-1]
                if cur_y==1:
                    c1_data[0].append(cur_i1)
                    c1_data[1].append(cur_i2)
                else:
                    c0_data[0].append(cur_i1)
                    c0_data[1].append(cur_i2)

            plt.xticks(np.arange(0.0,1.1,0.1))
            plt.yticks(np.arange(0.0,1.1,0.1))
            plt.xlim(0,1.05)
            plt.ylim(0,1.05)

            c0s = plt.scatter(c0_data[0],c0_data[1],s=40.0,c='r',label='Class -1')
            c1s = plt.scatter(c1_data[0],c1_data[1],s=40.0,c='b',label='Class 1')

            plt.legend(fontsize=10,loc=1)
            plt.show()
            return
		    

        print("Matrix dimensions not covered.")
    def train_weights(self):
        for i in range(self.epoch):
            processing_frame_text.insert(END, f"Epoch:{i}\n weight:{self.weights} \n")
            # self.plot(self.inputs, self.weights, f"Epoch {i}")
            for j in range(len(self.inputs)): 
                processing_frame_text.insert(END,f"Training on data at index {j}...\n")
                actual_output = self.get_weighted_sum(self.inputs[j],self.weights)
                error = self.desired_outputs[j]-actual_output
                processing_frame_text.insert(END, f"Error: {error} \n")
                for k in range(len(self.weights)):
                    previous_weight = self.weights[k]
                    self.weights[k] = self.weights[k] + self.lr*error*self.inputs[j][k]
                    processing_frame_text.insert(END,f"\tWeight[{k}]: {previous_weight} --->  {self.weights[k]} \n")
            processing_frame_text.insert(END,"next iteration \n")
        return self.weights



#Functions

def linearly_separable(perceptron, final_weights, desired_outputs,type):
    output_from_final_weight = 0
    linearly_separable = True
    result_frame_text.insert(END, f"Final Prediction Started\n")
    result_frame_text.insert(END, f"Final weight after Training: {final_weights}\n")
    for i in range(len(inputs)): 
        output_from_final_weight =  perceptron.get_weighted_sum(final_weights,inputs[i])
        result_frame_text.insert(END, f"\tPre-activation of Inputs {inputs[i][1:]} after computing with final weight ---> {final_weights.dot(inputs[i])}\n")
        if output_from_final_weight == desired_outputs[i]:
            result_frame_text.insert(END, f"\tWe obtain {output_from_final_weight} when we use our activation function. which is exactly {inputs[i][1:2]} {type} {inputs[i][2:]}\n")
        if output_from_final_weight != desired_outputs[i]:
            result_frame_text.insert(END, f"\tWe obtain {output_from_final_weight} when we use our activation function. which is not exactly {inputs[i][1:2]} {type} {inputs[i][2:]}\n")
            linearly_separable = False
    if linearly_separable:
        result_frame_text.insert(END,"This is a Linearly Separable Problem \n")
    else:
        result_frame_text.insert(END,"This is not a Linearly Separable Problem \n")


def button_clear():
    processing_frame_text.delete('1.0',END)
    result_frame_text.delete('1.0',END)
    return

def button_and():
    processing_frame_text.delete('1.0',END)
    result_frame_text.delete('1.0',END)

    desired_outputs = np.array([0,0,0,1])
    weights= np.array([0, 0, 0])
    processing_frame_text.insert(END, "[INFO] training perceptron..." + "\n")
    p = Perceptron(epoch,lr,inputs,weights,desired_outputs)
    final_weights = p.train_weights()
    linearly_separable(p, final_weights, desired_outputs,"AND")

    

        
def button_or():
    processing_frame_text.delete('1.0',END)
    result_frame_text.delete('1.0',END)
    
    desired_outputs = np.array([0,1,1,1])
    weights= np.array([0, 0, 0])
    processing_frame_text.insert(END, "[INFO] training perceptron..." + "\n")
    p = Perceptron(epoch,lr,inputs,weights,desired_outputs)
    final_weights = p.train_weights()
    linearly_separable(p, final_weights, desired_outputs,"OR")


def button_xor():
    processing_frame_text.delete('1.0',END)
    result_frame_text.delete('1.0',END)
    
    desired_outputs = np.array([0,1,1,0])
    weights= np.array([0, 0, 0])
    processing_frame_text.insert(END, "[INFO] training perceptron..." + "\n")
    p = Perceptron(epoch,lr,inputs,weights,desired_outputs)
    final_weights = p.train_weights()
    linearly_separable(p, final_weights, desired_outputs,"XOR")


        


processing_frame =LabelFrame(root, text="Processing", fg="Blue", width=800, height=200, padx=100, pady=10,font="Arial 20 bold italic")

button_frame =LabelFrame(root, text="Runnable Functions", fg="Blue", width=500, height=200, padx=115, pady=10, font="Arial 15 bold italic")

result_frame =LabelFrame(root, text="Conclusions", fg="Blue", width=800, height=200, padx=115, pady=10, font="Arial 20 bold italic")

processing_frame_text = Text(master=processing_frame)
scr=Scrollbar(processing_frame, orient=VERTICAL, command=processing_frame_text.yview)
scr.grid(row=0, column=1, rowspan=15, columnspan=1, sticky=NS)
processing_frame_text.grid(row=0, column=0, sticky=N)
processing_frame_text.config(yscrollcommand=scr.set, font=('Arial', 10, 'bold', 'italic'), background=('SystemButtonFace'), highlightthickness=0, border=0)

result_frame_text = Text(master=result_frame)
scr=Scrollbar(processing_frame, orient=VERTICAL, command=result_frame_text.yview)
scr.grid(row=0, column=1, rowspan=15, columnspan=1, sticky=NS)
result_frame_text.grid(row=0, column=0, sticky=N)
result_frame_text.config(yscrollcommand=scr.set, font=('Arial', 10, 'bold', 'italic'), background=('SystemButtonFace'), highlightthickness=0, border=0)


button_and = Button(button_frame, text="AND function",padx=15, pady=10 ,fg="White", bg="Blue",font="Arial 12", command= button_and)
button_or = Button(button_frame, text="OR function",padx=15, pady=10, fg="White", bg="Blue",font="Arial 12", command= button_or)
button_xor = Button(button_frame, text="XOR function",padx=15, pady=10, fg="White", bg="Blue",font="Arial 12", command= button_xor)
button_clear = Button(button_frame, text="Clear", fg="White", bg="Blue", padx=79, pady=20,font="Arial 12", command= button_clear)


# Arranging inside the root
# processing_frame.grid(row=0, column=0)

processing_frame.grid(row=0, column=0, columnspan=5, padx=10, pady=10)
button_frame.grid(row=0, column=6, columnspan=5, padx=10, pady=10)
button_and.grid(row=0,column=1, padx=5)
button_or.grid(row=0,column=2, padx=5)
button_xor.grid(row=0,column=3, padx=5)
button_clear.grid(row=1,column=0,columnspan=3, padx=5, pady=10)

result_frame.grid(row=6, column=0, columnspan=5, padx=10, pady=10)
root.mainloop()