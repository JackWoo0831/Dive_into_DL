'''
NMS:
'''
#In[]
import torch 

#定义计算IoU函数
def box_iou(boxes1,boxes2):
    '''
    input:
    boxes1,boxes2:输入的锚框,每个boxes的shape(box_number,4),其中
    4代表:xmin,ymin,xmax,ymax
    output:boxes1和boxes2中两两框的iou torch.Tensor
    注意:两两框的意思是假设boxes1有m个,boxes2有n个,则计算结果应是mn个.
    '''
    #计算每个box的面积
    box_area = lambda boxes : ((boxes[:,2] - boxes[:,0]) * 
                                (boxes[:,3] - boxes[:,1])) #box_area:(box_number,)

    area1 , area2 = box_area(boxes1), box_area(boxes2)

    #计算对应锚框的交集部分
    inter_upleft = torch.max(boxes1[:,None,:2],boxes2[:,:2]) #交集部分左上角坐标,就是比较两个框的(x_min,y_min)的较大者
    #特别注意:None的用法是扩充维度,也就是对每一个boxes1中的框,将所有的boxes2进行比较,下同
    inter_downright = torch.min(boxes1[:,None,2:],boxes2[:,2:]) #同理 比较(x_max,y_max)的较小者
    inters = (inter_downright - inter_upleft).clamp(min=0) #clamp是筛选大于0的值,小于0的输出0

    inter_area = inters[:,:,0] * inters[:,:,1] #交集部分面积 shape(box_number1,box_number2) 下同

    union_area = area1[:,None] + area2 - inter_area #并集面积 None扩充维度,为了维度一直相除

    return inter_area / union_area 


#定义NMS
def NMS(boxes,scores,threshold):
    '''
    input:
    boxes:anchor boxes,shape(number,4)
    scores:置信度 (numbers,)
    threshold:阈值 float
    output:筛选出的anchor box的index
    '''
    #先按置信度大小对所有锚框排序 降序 返回索引值
    B  = torch.argsort(scores,dim=-1,descending=True)

    keep = [] #要保留的box的index

    while B.numel() > 0: #还没筛完
        i = B[0] #选定当前置信度最高的
        keep.append(i) #加入
        
        if B.numel() == 1:
            break #只剩一个就完毕

        iou = box_iou(boxes[i,:].reshape(-1,4),
                        boxes[B[1:],:].reshape(-1,4)).reshape(-1) #计算选定框和剩余框的iou,reshape(-1)相当于flatten

        indexs = torch.nonzero(iou <= threshold).reshape(-1) #iou <= threshold是bool值,
        # 返回的是为True的值得索引,也就是满足小于阈值的索引

        B = B[indexs + 1] # 相当于是B = B[indexs],是筛选,然后将索引都加1,表示移动,删除掉已经计算完的基准框:B = B[indexs + 1]

    return torch.tensor(keep) 

        

#In[]
#test




                



        







    
# %%
