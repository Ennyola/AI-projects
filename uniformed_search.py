from tkinter import *
import collections
import time
root=Tk()
root.geometry("1200x600")

myCanvas = Canvas(root, width=1200,height=900,bg="white")
myCanvas.grid(row=1, column=0)

# root.configure(background="black")
# root.attributes("-topmost", True)
x_first=600
y_first=50
vals="ABCDEFGHI"
circles=dict()
j=0
visited = set() 
depth = 0

graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F', 'G'],
  'D' : ['H','I'],
  'E' : [],
  'F' : [],
  'G' :[],
  'H' : [],
  'I' : []
}



def create_circle(x,y,r,canvasName):
    x0=x-r
    y0=y-r
    x1=x+r
    y1=y+r
    return canvasName.create_oval(x0,y0,x1,y1,fill="#BB6BD9")
def line(x1,y1,x2,y2,canvasName):
    return canvasName.create_line(x1,y1,x2,y2,fill="#BB6BD9",width=2,smooth=True)
c=x_first
d=y_first
cf=1/(len(vals)-1)
def create_tree(c,d,i=0,mc=1.2):
    global vals,myCanvas,root,cf,j
    if i<len(vals):
        z=create_circle(c,d,20,myCanvas)
        root.update()
        m=Label(root,text=vals[i],bg="#BB6BD9",fg="white")
        m.config(font=("courier 16 bold"))
        m.place(x=c-8,y=d-12)
        circles[vals[i]]=(z,m)
        root.update()
        #time.sleep(1)
        if 2*i+1 <len(vals):
            x=line(c,d+20,c-100*mc,d+100,myCanvas)
            root.update()
            #time.sleep(0.5)
        create_tree(c-(100*mc),d+100,2*i+1,mc-mc*cf)
        if 2*i+2<len(vals):
            x=line(c,d+20,c+100*mc,d+100,myCanvas)
            root.update()
            #time.sleep(0.5)
        create_tree(c+(100*mc),d+100,2*i+2,mc-mc*2*cf)
        root.update()
create_tree(c,d)

class tree:
    def __init__(self,val,i=0):
        if i<len(val):
            self.val=val[i]
            self.left=tree(val,2*i+1)
            self.right=tree(val,2*i+2)
        else:
            self.left=None
            self.right=None
        
    def depth_first_search(self,level=0):
        global inorder,circles,myCanvas,root,xc
        if self.left is not None and self.right is not None:
            c=circles[self.val]
            myCanvas.itemconfig(c[0],fill="#6D6BD9")
            c[1].configure(bg="#6D6BD9",fg="white")
            m=Label(root,text=self.val,fg="#6D6BD9",bg="white",font="courier 15 bold")
            m.place(x=xc,y=460)
            root.update()
            time.sleep(0.5)
            xc+=30
            self.left.depth_first_search(level+1)
            self.right.depth_first_search(level+1)
            
    def BFS(self, graph, node):
        global circles,myCanvas,root,xc
        visites = [] # List to keep track of visited nodes.
        queue = []  

        visites.append(node)
        queue.append(node)
        
        while queue:
            s = queue.pop(0) 
            c=circles[s]
            myCanvas.itemconfig(c[0],fill="#6D6BD9")
            c[1].configure(bg="#6D6BD9",fg="white")
            m=Label(root,text=s,fg="#6D6BD9",bg="white",font="courier 15 bold")
            m.place(x=xc,y=460)
            root.update()
            time.sleep(0.5)
            xc+=30

            for neighbour in graph[s]:
                if neighbour not in visited:
                    visites.append(neighbour)
                    queue.append(neighbour)
                    
                    
    def depth_limited_search(self,graph, node,goal, limit):
        global depth,circles,myCanvas,root,xc
        queue = []  
        queue.append(node)
        if(depth <= limit):
            s = queue.pop(0)
            if (s==goal):
                c=circles[s]
                myCanvas.itemconfig(c[0],fill="#6D6BD9")
                c[1].configure(bg="#6D6BD9",fg="white")
                m=Label(root,text=s,fg="#6D6BD9",bg="white",font="courier 15 bold")
                m.place(x=xc,y=460)
                root.update()
                time.sleep(0.5)
                xc+=30
                visited.add(node)
            if node not in visited:
                c=circles[s]
                myCanvas.itemconfig(c[0],fill="#6D6BD9")
                c[1].configure(bg="#6D6BD9",fg="white")
                m=Label(root,text=s,fg="#6D6BD9",bg="white",font="courier 15 bold")
                m.place(x=xc,y=460)
                root.update()
                time.sleep(0.5)
                xc+=30
                visited.add(node)
                depth +=1
                for neighbour in graph[node]:
                    print(neighbour)
                    self.depth_limited_search(graph, neighbour, goal, limit)
        else:
            print("I no see am ohhh")
    


tr=tree(vals)
m=Label(root,text="Traversal:",fg="#6D6BD9",bg="white",font="courier 15 bold")
m.place(x=10,y=460)
xc=160
# tr.depth_first_search()
# tr.BFS(graph,'A')
tr.depth_limited_search(graph,'A','H',2)
root.mainloop()
