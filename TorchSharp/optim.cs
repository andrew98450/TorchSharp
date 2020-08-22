using NumSharp;
using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Reflection;

namespace TorchSharp
{
    namespace optim
    {
        public class SGD
        {
            public List<nn.Module> parameter;

            double lr;

            public SGD(List<nn.Module> parameter, double lr)
            {
                this.parameter = parameter;
                this.lr = lr;
            }
            public void zero_grad() 
            {
                for (int i = 0; i < parameter.Count; i++) 
                {
                    parameter[i].weight.grad = np.zeros(parameter[i].weight.grad.Shape);
                    parameter[i].bias.grad = np.zeros(parameter[i].bias.grad.Shape);
                }
            }
            public void step() 
            {
                for (int i = 0; i < parameter.Count; i++)
                {
                    parameter[i].weight.data = parameter[i].weight.data - (lr * parameter[i].weight.grad);
                    parameter[i].bias.data = parameter[i].bias.data - (lr * parameter[i].bias.grad);
                }
            }
        }
        public class Adam
        { 
            
        }
    }
}
