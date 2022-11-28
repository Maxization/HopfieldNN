using System;
using System.Collections.Generic;
using System.Formats.Tar;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace HopfieldNN
{
    public enum LearningRule
    {
        Oja,
        Hebb,
    }
    public class HopfieldNetwork
    {
        private int _neuronCount;
        [JsonProperty("learning_rate")]
        private double _lr;
        [JsonProperty("rule")]
        private LearningRule _rule;
        private int _ojaMaxIters;
        [JsonProperty("weights")]
        private double[,] _weights;
        private double _threshold;
        private Random _rng;
        int _width, _height;
        int _predicted = 0;
        bool _saveBitmaps;

        public HopfieldNetwork(LearningRule rule, int width, int height, double lr=1e-7, int ojaMaxIters=100, int seed = 42, bool saveBitmaps = false)
        {
            _threshold = 0;
            _neuronCount = width * height;
            _lr = lr;
            _rule = rule;
            _weights = new double[_neuronCount, _neuronCount];
            _rng = new Random(seed);
            _ojaMaxIters = ojaMaxIters;
            _width = width;
            _height = height;
            _saveBitmaps = saveBitmaps;
        }

        public void Train(int[][] trainingData)
        {
            switch (_rule)
            {
                case LearningRule.Oja:
                    trainOja(trainingData);
                    break;
                case LearningRule.Hebb:
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
                Console.WriteLine($"Oja iteration: {q}");

                var old = (double[,])_weights.Clone();

                foreach(var pattern in trainingData)
                {
                    var V = new double[pattern.Length];

                    for (int i = 0; i < pattern.Length; i++)
                    {
                        var sum = 0.0;
                        for (int j = 0; j < pattern.Length; j++)
                        {
                            sum += old[i, j] * pattern[j];
                        }

                        V[i] = sum;
                    }

                    for (int i = 0; i < _neuronCount; i++)
                    {
                        for (int j = 0; j < _neuronCount; j++)
                        {
                            _weights[i, j] += _lr * V[i] * (pattern[j] - V[i] * old[i, j]);
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
                Console.WriteLine(k);
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
            var maxIt = 100;
            var iterations = 0;
            var input = (int[])_input.Clone();

            var energy = Energy(input);

            if (synch)
            {
                var output = new int[input.Length];
                while (iterations < maxIt)
                {
                    if (_saveBitmaps)
                    {
                        DataHelper.CreateBitmap(input, _height, _width, _predicted, "out", "_" + iterations.ToString());
                    }

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
                        if (_saveBitmaps)
                        {
                            DataHelper.CreateBitmap(input, _height, _width, _predicted, "out", "_" + (iterations + 1).ToString());
                        }
                        _predicted++;
                        return input;
                    }

                    energy = newEnergy;
                    iterations++;
                }
            }
            else
            {
                while (iterations < maxIt)
                {
                    if (_saveBitmaps)
                    {
                        DataHelper.CreateBitmap(input, _height, _width, _predicted, "out", "_" + iterations.ToString());
                    }

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
                        Console.WriteLine($"Prediction iterations: {iterations}");
                        if (_saveBitmaps)
                        {
                            DataHelper.CreateBitmap(input, _height, _width, _predicted, "out", "_" + (iterations + 1).ToString());
                        }
                        _predicted++;
                        return input;
                    }

                    energy = newEnergy;
                    iterations++;
                }
            }

            _predicted++;
            if (_saveBitmaps)
            {
                DataHelper.CreateBitmap(input, _height, _width, _predicted, "out", "_" + (iterations + 1).ToString());
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

        private double[] Multiplication(double[,] x, int[] y)
        {
            var result = new double[x.GetLength(0)];
            for (int i = 0; i < x.GetLength(0); i++)
            {
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    result[i] += x[i, j] * y[j];
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
