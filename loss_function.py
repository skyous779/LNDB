from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class Binary_Loss(nn.Module):
    def __init__(self):
        super(Binary_Loss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.BCELoss()



    def forward(self, model_output, targets):
        #targets[targets == 0] = -1

        # torch.empty(3, dtype=torch.long)
        # model_output = model_output.long()
        # targets = targets.long()
        # print(model_output)
        # print(F.sigmoid(model_output))
        # print(targets)
        # print('kkk')
        # model_output =torch.LongTensor(model_output.cpu())
        # targets =torch.LongTensor(targets.cpu())
        # model_output = model_output.type(torch.LongTensor)
        # targets = targets.type(torch.LongTensor)
        loss = self.criterion(model_output, targets)

       
        return loss





def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        


        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
 
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
 
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



# class Focal_loss(nn.Module):

#     def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):

#         super(focal_loss, self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha, (float, int)):    #仅仅设置第一类别的权重
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1 - alpha)
#         if isinstance(alpha, list):  #全部权重自己设置
#             self.alpha = torch.Tensor(alpha)
#         self.gamma = gamma


#     def forward(self, inputs, targets):
#         alpha = self.alpha
#         print('aaaaaaa',alpha)
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs,dim=1)
#         print('ppppppppppppppppppppp', P)
#         # ---------one hot start--------------#
#         class_mask = inputs.data.new(N, C).fill_(0)  # 生成和input一样shape的tensor
#         print('依照input shape制作:class_mask\n', class_mask)
#         class_mask = class_mask.requires_grad_()  # 需要更新， 所以加入梯度计算
#         ids = targets.view(-1, 1)  # 取得目标的索引
#         print('取得targets的索引\n', ids)
#         class_mask.data.scatter_(1, ids.data, 1.)  # 利用scatter将索引丢给mask
#         print('targets的one_hot形式\n', class_mask)  # one-hot target生成
#         # ---------one hot end-------------------#
#         probs = (P * class_mask).sum(1).view(-1, 1)
#         print('留下targets的概率（1的部分），0的部分消除\n', probs)
#         # 将softmax * one_hot 格式，0的部分被消除 留下1的概率， shape = (5, 1), 5就是每个target的概率

#         log_p = probs.log()
#         print('取得对数\n', log_p)
#         # 取得对数
#         loss = torch.pow((1 - probs), self.gamma) * log_p
#         batch_loss = -alpha *loss.t()  # 對應下面公式
#         print('每一个batch的loss\n', batch_loss)
#         # batch_loss就是取每一个batch的loss值

#         # 最终将每一个batch的loss加总后平均
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         print('loss值为\n', loss)
#         return loss

# class focal_loss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):
#         """
#         focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
#         步骤详细的实现了 focal_loss损失函数.
#         :param alpha:   阿尔法α,类别权重. 当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.255
#         :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
#         :param num_classes:     类别数量
#         :param size_average:    损失计算方式,默认取均值
#         """
#         super(focal_loss,self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha,list):
#             assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
#             print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
#             self.alpha = torch.Tensor(alpha)
#         else:
#             assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
#             print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

#         self.gamma = gamma

#     def forward(self, preds, labels):
#         """
#         focal_loss损失计算
#         :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B批次, N检测框数, C类别数
#         :param labels:  实际类别. size:[B,N] or [B]        [B*N个标签(假设框中有目标)]，[B个标签]
#         :return:
#         """
#         assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"        
#         preds = preds.contiguous().view(preds.shape[0], -1)
#         labels = labels.contiguous().view(labels.shape[0], -1)




#         #固定类别维度，其余合并(总检测框数或总批次数)，preds.size(-1)是最后一个维度
#         preds = preds.view(-1,2)   #n*2
#         self.alpha = self.alpha.to(preds.device)
        
#         #使用log_softmax解决溢出问题，方便交叉熵计算而不用考虑值域
#         preds_logsoft = F.log_softmax(preds, dim=1) 
        
#      	#log_softmax是softmax+log运算，那再exp就算回去了变成softmax
#         preds_softmax = torch.exp(preds_logsoft)    
    
#         labels= labels.to(torch.int64)
#         # 这部分实现nll_loss ( crossentropy = log_softmax + nll)

#         preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
#         preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))

#         self.alpha = self.alpha.gather(0,labels.view(-1)) 

#         # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        
#         #torch.mul 矩阵对应位置相乘，大小一致
#         loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 
    
#         #torch.t()求转置
#         loss = torch.mul(self.alpha, loss.t())
#         #print(loss.size()) [1,5]
        
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
       
#         return loss
