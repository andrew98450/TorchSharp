using NumSharp;
using System;
using System.Collections.Generic;
namespace TorchSharp
{
    namespace nn
    {
        class Init
        {
            public static Tensor xavier_uniform(Tensor tensor, double gain)
            {
                double[] fans = calc_fan_in_and_fan_out(tensor);
                double a = gain * Math.Sqrt(6 / (fans[0] + fans[1]));
                NDArray array = np.random.uniform(-a, a, tensor.data.shape);
                Tensor tensors = new Tensor(array);
                return tensors;
            }
            public static Tensor xavier_normal(Tensor tensor, double gain)
            {
                double[] fans = calc_fan_in_and_fan_out(tensor);
                double std = gain * Math.Sqrt(2 / (fans[0] + fans[1]));
                NDArray array = np.random.normal(0, Math.Pow(std, 2.0), tensor.data.shape);
                Tensor tensors = new Tensor(array);
                return tensors;
            }
            public static Tensor kaiming_uniform(Tensor tensor, double a, string mode, string nonlinearity)
            {
                double[] fans = calc_fan_in_and_fan_out(tensor);
                double gain = calculate_gain(nonlinearity, a);
                double bound = 0;
                if (mode == "fan_in")
                    bound = gain * Math.Sqrt(3 / fans[0]);
                else
                    bound = gain * Math.Sqrt(3 / fans[1]);
                NDArray array = np.random.uniform(-bound, bound, tensor.data.shape);
                Tensor tensors = new Tensor(array);
                return tensors;
            }
            public static Tensor kaiming_normal(Tensor tensor, double a, string mode, string nonlinearity)
            {
                double[] fans = calc_fan_in_and_fan_out(tensor);
                double gain = calculate_gain(nonlinearity, a);
                double std;
                if (mode == "fan_in")
                    std = gain / Math.Sqrt(fans[0]);
                else
                    std = gain / Math.Sqrt(fans[1]);
                NDArray array = np.random.normal(0, Math.Pow(std, 2.0), tensor.data.shape);
                Tensor tensors = new Tensor(array);
                return tensors;
            }
            public static Tensor uniform(Tensor tensor, double a, double b)
            {
                NDArray array = np.random.uniform(a, b, tensor.data.shape);
                Tensor tensors = new Tensor(array);
                return tensors;
            }
            public static Tensor normal(Tensor tensor, double mean, double std)
            {
                NDArray array = np.random.normal(mean, std, tensor.data.shape);
                Tensor tensors = new Tensor(array);
                return tensors;
            }
            public static double calculate_gain(string nonlinearity, double parm)
            {
                double result = 0;
                switch (nonlinearity)
                {
                    case "linear":
                        result = 1;
                        break;
                    case "conv":
                        result = 1;
                        break;
                    case "sigmoid":
                        result = 1;
                        break;
                    case "tanh":
                        result = 5 / 3;
                        break;
                    case "relu":
                        result = Math.Sqrt(2.0);
                        break;
                    case "leaky_relu":
                        result = Math.Sqrt(2 / (1 + Math.Pow(parm, 2.0)));
                        break;
                }
                return result;
            }
            public static double[] calc_fan_in_and_fan_out(Tensor tensor)
            {
                double[] fans = new double[2];
                int[] shape = tensor.data.shape;
                if (shape.Length == 2)
                {
                    fans[0] = shape[1];
                    fans[1] = shape[0];
                }
                else
                {
                    fans[0] = shape[1];
                    fans[1] = shape[0];
                    double size = 1;
                    if (shape.Length > 2)
                    {
                        int[] numel = tensor.data[0, 0].shape;
                        for (int i = 0; i < numel.Length; i++)
                            size *= numel[i];
                    }
                    fans[0] *= size;
                    fans[1] *= size;
                }
                return fans;
            }
        }
        
        public class Module 
        {
            public Tensor weight = new Tensor(np.zeros());
            public Tensor bias = new Tensor(np.zeros());
            public static List<Module> parameter = new List<Module>();
            public virtual void register_parameter(Module module)
            {
                parameter.Add(module);
            }
            public virtual List<Module> parameters() 
            {
                return parameter;
            }
            public virtual Tensor forward(Tensor input) 
            {
                for (int i = 0; i < parameter.Count; i++) 
                    input = parameter[i].forward(input);
                return input;
            }
        }
        public class Sequential : Module 
        {
            public Sequential(params Module[] module) 
            {
                for (int i = 0; i < module.Length; i++)
                {
                    register_parameter(module[i]);
                }
            }
            public override List<Module> parameters()
            {
                return parameter;
            }
            public override Tensor forward(Tensor input)
            {
                for (int i = 0; i < parameter.Count; i++)
                    input = parameter[i].forward(input);
                return input;
            }
        }
        public class Linear : Module
        {
            int in_feat;
            int out_feat;
            
            public Linear(int in_feat, int out_feat) 
            {
                this.in_feat = in_feat;
                this.out_feat = out_feat;
                reset_parameter();
            }
            public void reset_parameter() 
            {
                Tensor weight_array = new Tensor(np.zeros(out_feat, in_feat));
                Tensor bias_array = new Tensor(np.zeros(out_feat));
                double[] fan =  Init.calc_fan_in_and_fan_out(weight_array);
                double bound = 1 / Math.Sqrt(fan[0]);
                weight = Init.kaiming_uniform(weight_array, Math.Sqrt(5), "fan_in", "relu");
                bias = Init.uniform(bias_array, -bound, bound);
            }
            public override Tensor forward(Tensor input) 
            {
                return Tensor.linear(input, weight, bias);
            }
        }
        public class ReLU : Module
        {
            public override Tensor forward(Tensor input)
            {
                return Tensor.relu(input);
            }
        }
        public class MSELoss : Module
        {
            public Tensor forward(Tensor output, Tensor label)
            {
                return Tensor.mse_loss(output, label);
            }
        }
    }
}
