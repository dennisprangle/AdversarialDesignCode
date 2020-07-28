import torch
from torch.distributions import LogNormal
import numpy as np
import matplotlib.pyplot as plt
import advOpt as adv
plt.ion()

class PK_FIM(adv.FIM):
  def __init__(self, nsamples):
    self.npars = 2
    self.nsamples = nsamples
    loc = torch.tensor(np.log((0.1, 1., 20.)), dtype=torch.float32)
    scale = torch.tensor(np.sqrt((0.05, 0.05, 0.05)), dtype=torch.float32)
    self.prior = LogNormal(loc, scale)

  def estimate_FIM(self, theta, design):
    # Unvectorised code for reference (theta is a vector)
    # x = 400. * theta[1] * (torch.exp(-theta[0]*design) - torch.exp(-theta[1]*design)) / (theta[2]*(theta[1]-theta[0]))
    # grad0 = x/(theta[1]-theta[0]) - design*400.*theta[1]/(theta[2]*(theta[1]-theta[0]))*torch.exp(-theta[0]*design)
    # grad1 = x/theta[1] - x/(theta[1]-theta[0]) + design*400.*theta[1]*torch.exp(-theta[1]*design)/(theta[2]*(theta[1]-theta[0]))
    # grad2 = -x/theta[2]
    # jacobian = torch.stack((grad0, grad1, grad2), dim=1)
    # fim = torch.mm(jacobian.transpose(0,1), jacobian)
    design = design.unsqueeze(0)
    theta1 = theta[:, 0:1]
    theta2 = theta[:, 1:2]
    theta3 = theta[:, 2:3]
    x = 400. * theta2 * (torch.exp(-theta1*design) - torch.exp(-theta2*design)) / (theta3*(theta2-theta1))    
    grad0 = x/(theta2-theta1) - design*400.*theta2/(theta3*(theta2-theta1))*torch.exp(-theta1*design)
    grad1 = x/theta2 - x/(theta2-theta1) + design*400.*theta2*torch.exp(-theta2*design)/(theta3*(theta2-theta1))
    grad2 = -x/theta3
    jacobian = torch.stack((grad0, grad1, grad2), dim=2)
    fim = torch.matmul(jacobian.transpose(1,2), jacobian)
    return fim

  
exampleFIM = PK_FIM(nsamples=100)
dd = torch.tensor((1.0,2.0,3.0,4.0))
AA = torch.eye(3)
exampleFIM.estimateK(dd, AA)
