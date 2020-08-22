using NumSharp;
using System;
using System.Collections.Generic;
namespace TorchSharp
{
    public class Tensor 
    {
        public NDArray data;
        public NDArray grad;
        public Dictionary<Tensor, NDArray> create;
        public static Stack<Dictionary<Tensor, NDArray>> grads;
        public Tensor()
        {
            data = np.zeros();
            create = new Dictionary<Tensor, NDArray>();
            grad = np.zeros();
            grads = new Stack<Dictionary<Tensor, NDArray>>();
        }
        public Tensor(NDArray data)
        {
            this.data = data;
            create = new Dictionary<Tensor, NDArray>();
            grad = np.zeros(this.data.Shape);
            grads = new Stack<Dictionary<Tensor, NDArray>>();
        }
        public static Tensor operator +(Tensor a, Tensor b)
        {
            return new AddBackward(a, b, grads);
        }
        public static Tensor operator +(Tensor a, double b)
        {
            return new AddBackward(a, b, grads);
        }
        public static Tensor operator -(Tensor a, Tensor b)
        {
            return new SubBackward(a, b, grads);
        }
        public static Tensor operator -(Tensor a, double b)
        {
            return new SubBackward(a, b, grads);
        }
        public static Tensor operator -(Tensor a)
        {
            return a * -1;
        }
        public static Tensor operator *(Tensor a, Tensor b)
        {
            return new MulBackward(a, b, grads);
        }
        public static Tensor operator *(Tensor a, double b)
        {
            return new MulBackward(a, b, grads);
        }
        public static Tensor operator /(Tensor a, Tensor b)
        {
            return new DivBackward(a, new InvBackward(b, grads), grads);
        }
        public static Tensor operator /(Tensor a, double b)
        {
            return new DivBackward(a, b, grads);
        }
        public static Tensor operator ^(Tensor a, double b)
        {
            return new PowBackward(a, b, grads);
        }
        public override string ToString()
        {
            return data.ToString();
        }
        public static Tensor sin(Tensor a)
        {
            return new SinBackward(a, grads);
        }
        public static Tensor cos(Tensor a)
        {
            return new CosBackward(a, grads);
        }
        public static Tensor tan(Tensor a)
        {
            return new TanBackward(a, grads);
        }
        public static Tensor tanh(Tensor a)
        {
            return new TanhBackward(a, grads);
        }
        public static Tensor sigmoid(Tensor a)
        {
            return new SigmoidBackward(a, grads);
        }
        public static Tensor relu(Tensor a)
        {
            return new ReLUBackward(a, grads);
        }
        public static Tensor exp(Tensor a)
        {
            return new ExpBackward(a, grads);
        }
        public static Tensor log(Tensor a)
        {
            return new LogBackward(a, grads);
        }
        public static Tensor mean(Tensor a)
        {
            return new MeanBackward(a, grads);
        }
        public static Tensor sum(Tensor a)
        {
            return new SumBackward(a, grads);
        }
        public static Tensor mm(Tensor a, Tensor b)
        {
            return new MMBackward(a, b, grads);
        }
        public static Tensor mse_loss(Tensor output, Tensor label) 
        {
            return mean((output - label) ^ 2);
        }
        public static Tensor linear(Tensor input, Tensor weight, Tensor bias)
        {
            return mm(input, new Tensor(weight.data)) + bias;
        }
        public void backward()
        {
            create = new Dictionary<Tensor, NDArray>();
            int count = 0, item = grads.Count;
            while (count <= item)
            {
                Dictionary<Tensor, NDArray> stack = grads.Pop();
                foreach (var node in stack.Keys)
                {
                    if (!create.ContainsKey(node))
                        create.Add(node, stack[node]);
                    else
                        create[node] += stack[node];

                    if (node.GetType() != typeof(Tensor))
                    {
                        Dictionary<Tensor, NDArray> child = new Dictionary<Tensor, NDArray>(node.create);
                        foreach (var child_node in node.create.Keys)
                        {
                            if (stack[node].shape.Length == 2 && node.create[child_node].shape.Length == 2)
                                if (stack[node].shape[1] == node.create[child_node].shape[0])
                                    child[child_node] = np.dot(stack[node], node.create[child_node]);
                                else
                                    child[child_node] = stack[node] * node.create[child_node];
                            else
                                child[child_node] = stack[node] * node.create[child_node];
                            grads.Push(child);
                        }
                    }
                }
                stack.Clear();
                count++;
            }
            foreach (var node in create.Keys)
            {
                node.grad = create[node];
            }
            grads.Clear();
        }

        class AddBackward : Tensor
        {
            public AddBackward(Tensor a, Tensor b, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = a.data + b.data;
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.ones(a.data.Shape));
                create.Add(b, np.ones(b.data.Shape));
                Tensor.grads.Push(create);
            }
            public AddBackward(Tensor a, double b, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = a.data + b;
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.ones(a.data.Shape));
                Tensor.grads.Push(create);
            }
        }
        class MulBackward : Tensor
        {
            public MulBackward(Tensor a, Tensor b, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = a.data * b.data;
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, b.data);
                create.Add(b, a.data);
                Tensor.grads.Push(create);
            }
            public MulBackward(Tensor a, double b, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = a.data * b;
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.zeros(a.data.Shape) + b);
                Tensor.grads.Push(create);
            }
        }
        class MMBackward : Tensor
        {
            public MMBackward(Tensor a, Tensor b, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.matmul(a.data, b.data.transpose());
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.matmul(b.data, np.ones(a.data.transpose().Shape)));
                create.Add(b, np.matmul(a.data, np.ones(b.data.transpose().Shape)));
                Tensor.grads.Push(create);
            }
        }
        class PowBackward : Tensor
        {
            public PowBackward(Tensor a, double b, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.power(a.data, b);
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.power(a.data, b - 1) * b);
                Tensor.grads.Push(create);
            }
        }

        class DivBackward : Tensor
        {
            public DivBackward(Tensor a, Tensor b, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = a.data / b.data;
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, b.data);
                create.Add(b, a.data);
                Tensor.grads.Push(create);
            }
            public DivBackward(Tensor a, double b, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = a.data / b;
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, a.data);
                Tensor.grads.Push(create);
            }
        }
        class SubBackward : Tensor
        {
            public SubBackward(Tensor a, Tensor b, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = a.data - b.data;
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.ones(a.data.Shape));
                create.Add(b, np.ones(b.data.Shape) * -1);
                Tensor.grads.Push(create);
            }
            public SubBackward(Tensor a, double b, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = a.data - b;
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.ones(a.data.Shape));
                Tensor.grads.Push(create);
            }
        }
        class SinBackward : Tensor
        {
            public SinBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.sin(a.data);
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.cos(a.data));
                Tensor.grads.Push(create);
            }
        }
        class CosBackward : Tensor
        {
            public CosBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.cos(a.data);
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, -np.sin(a.data));
                Tensor.grads.Push(create);
            }
        }
        class TanBackward : Tensor
        {
            public TanBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.tan(a.data);
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.power(1 / np.cos(a.data), 2));
                Tensor.grads.Push(create);
            }
        }
        class SigmoidBackward : Tensor
        {
            public SigmoidBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = 1 / (1 + np.exp(-a.data));
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, data * (1 - data));
                Tensor.grads.Push(create);
            }
        }
        class TanhBackward : Tensor
        {
            public TanhBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.tanh(a.data);
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, 1 - np.power(data, 2));
                Tensor.grads.Push(create);
            }
        }
        class ReLUBackward : Tensor
        {
            public ReLUBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.maximum(a.data, 0);
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.maximum(np.sign(a.data), 0));
                Tensor.grads.Push(create);
            }
        }
        class MeanBackward : Tensor
        {
            public MeanBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.mean(a.data);
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.mean(a.data / data) / a.data.flatten().shape[0]);
                Tensor.grads.Push(create);
            }
        }
        class SumBackward : Tensor
        {
            public SumBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.sum(a.data);
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, np.sum(a.data / data));
                Tensor.grads.Push(create);
            }
        }
        class ExpBackward : Tensor
        {
            public ExpBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.exp(a.data);
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, data);
                Tensor.grads.Push(create);
            }
        }
        class LogBackward : Tensor
        {
            public LogBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = np.log(a.data);
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, 1 / a.data);
                Tensor.grads.Push(create);
            }
        }
        class InvBackward : Tensor
        {
            public InvBackward(Tensor a, Stack<Dictionary<Tensor, NDArray>> grads)
            {
                data = 1 / a.data;
                create = new Dictionary<Tensor, NDArray>();
                Tensor.grads = grads;
                create.Add(a, -np.power(a.data, -2));
                Tensor.grads.Push(create);
            }
        }
    }
}
