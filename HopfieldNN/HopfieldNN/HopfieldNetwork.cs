using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace HopfieldNN
{

    public class HopfieldNetwork
    {
        private int _neuronCount;
        [JsonProperty("learning_rate")]
        private double _lr;
        [JsonProperty("rule")]
        private string _rule;
        private int _ojaMaxIters;
        [JsonProperty("weights")]
        private double[,] _weights;
        private double _threshold;
        private Random _rng;
        public HopfieldNetwork(int neuronCount, string rule, double lr=10e-7, int ojaMaxIters=100)
        {
            _threshold = 0;
            _neuronCount = neuronCount;
            _lr = lr;
            _rule = rule;
            _weights = new double[neuronCount, neuronCount];
            _rng = new Random(42);
            _ojaMaxIters = ojaMaxIters;
        }

        public void Train(int[][] trainingData)
        {
            switch (_rule)
            {
                case "oja":
                    trainOja(trainingData);
                    break;
                case "hebb":
                    trainHebb(trainingData);
                    break;
                default:
                    throw new ArgumentException($"rule {_rule} is not supported");
            }
        }

        private void trainOja(int[][] trainingData)
        {
            trainHebb(trainingData);
            for (int q = 0; q < _ojaMaxIters; q++)
            {
                if (q % 3 == 0)
                {
                    Console.WriteLine($"Iteration: {q}");
                }

                var old = (double[,])_weights.Clone();
                for (int i = 0; i < _neuronCount; i++)
                {
                    for (int j = 0; j < _neuronCount; j++)
                    {
                        if (i == j)
                            continue;
                        foreach (var pattern in trainingData)
                        {
                            var V = 0.0;
                            for (int k = 0; k < pattern.Length; k++)
                            {
                                V += pattern[k] * _weights[i, k];
                            }
                            _weights[i, j] += _lr * V * (pattern[i] - V * _weights[i, j]);
                        }
                    }
                }
                
                if (diffNorm(old, _weights) < 1e-10)
                {
                    break;
                }
            }
        }

        private double diffNorm(double[,] a, double[,] b)
        {
            double norm = 0;
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                    norm += Math.Abs(a[i, j] - b[i, j]);
            }

            return norm;
        }

        private void trainHebb(int[][] trainingData)
        {
            int trainSize = trainingData.GetLength(0);
            for (int k = 0; k < trainSize; k++)
            {
                var input = trainingData[k];
                for (int i = 0; i < input.Length; i++)
                {
                    // Hebb rule
                    for (int j = i; j < input.Length; j++)
                    {
                        if (i == j)
                        {
                            continue;
                        }

                        var weight = input[i] * input[j];
                        _weights[i, j] += weight;
                        _weights[j, i] += weight;
                    }
                }
            }

            for (int i = 0; i < _neuronCount; i++)
            {
                for (int j = i; j < _neuronCount; j++)
                {
                    _weights[i, j] /= trainSize;
                    _weights[j, i] /= trainSize;
                }
            } 
        }

        public int[] Predict(int[] _input, bool synch = true)
        {
            var iterations = 0;
            var input = (int[])_input.Clone();

            var energy = Energy(input);

            if (synch)
            {
                var output = new int[input.Length];
                while (iterations < 100)
                {
                    //Synch
                    for (int i = 0; i < output.Length; i++)
                    {
                        var sum = 0.0;
                        for (int j = 0; j < output.Length; j++)
                        {
                            sum += _weights[i, j] * input[j];
                        }

                        sum -= _threshold;

                        if (sum >= 0)
                        {
                            output[i] = 1;
                        }
                        else
                        {
                            output[i] = -1;
                        }
                    }

                    for (int i = 0; i < input.Length; i++)
                    {
                        input[i] = output[i];
                    }

                    var newEnergy = Energy(input);

                    if (Math.Abs(newEnergy - energy) < 1e-8)
                    {
                        Console.WriteLine(iterations);
                        return input;
                    }

                    energy = newEnergy;
                    iterations++;
                }
            }
            else
            {
                while (iterations > 0)
                {
                    //Async
                    for (int k = 0; k < 100; k++)
                    {
                        var ind = _rng.Next(_neuronCount);

                        var sum = 0.0;
                        for (int j = 0; j < input.Length; j++)
                        {
                            sum += _weights[ind, j] * input[j];
                        }
                        sum -= _threshold;

                        if (sum >= 0)
                        {
                            input[ind] = 1;
                        }
                        else
                        {
                            input[ind] = -1;
                        }
                    }

                    var newEnergy = Energy(input);

                    if (Math.Abs(newEnergy - energy) < 1e-8)
                    {
                        Console.WriteLine(iterations);
                        return input;
                    }

                    energy = newEnergy;
                    iterations--;
                }
            }

            return input;
        }

        private double Energy(int[] data)
        {
            var v = Multiplication(data, _weights);
            double dotProduct = v.Zip(data, (d1, d2) => d1 * d2).Sum();
            return -0.5 * dotProduct + data.Sum(x => x * _threshold);
        }

        private double[] Multiplication(int[] x, double[,] y)
        {
            var result = new double[y.GetLength(1)];
            for (int i = 0; i < y.GetLength(1); i++)
            {
                for (int j = 0; j < y.GetLength(0); j++)
                {
                    result[i] += x[j] * y[j, i]; 
                }
            }

            return result;
        }

        public void Save(string name)
        {
            string json = JsonConvert.SerializeObject(this);
            var dir = "../../../../../trained";
            Directory.CreateDirectory(dir);
            File.WriteAllText($"{dir}/{name}.json", json);
        }
    }
}
