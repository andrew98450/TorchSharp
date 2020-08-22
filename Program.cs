using System;
using NumSharp;
using TorchSharp;
using TorchSharp.nn;
using TorchSharp.optim;
namespace AutoDiff
{
    class Program
    {
        static void Main(string[] args)
        {
            Sequential xornet = new Sequential(
                new Linear(2, 100),
                new ReLU(),
                new Linear(100, 1));

            double[,] x = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
            double[,] y = { { 0 }, { 1 }, { 1 }, { 0 } };

            NDArray a_np = np.array(x);
            NDArray b_np = np.array(y);

            Tensor input = new Tensor(a_np);
            Tensor label = new Tensor(b_np);

            int epoch = 1000;
            SGD optim = new SGD(xornet.parameters(), 0.05);
            MSELoss mse = new MSELoss();

            for (int i = 1; i <= epoch; i++)
            { 
                Tensor output = xornet.forward(input);
                Tensor loss = mse.forward(output, label);
                optim.zero_grad();
                loss.backward();
                optim.step();
                Console.WriteLine("[+] Epoch: " + i + " Loss: " + loss);
            }

            Tensor z = new Tensor(new NDArray(x));
            Tensor outputs = xornet.forward(z);
            Console.WriteLine("Result: " + outputs.data.flatten().ToString());
            Console.ReadLine();
        }
    }
}
