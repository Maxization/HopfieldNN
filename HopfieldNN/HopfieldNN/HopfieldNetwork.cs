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
        private Random _rng;
        public HopfieldNetwork(int neuronCount, string rule, double lr=10e-7, int ojaMaxIters=100)
        {
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
                var old = (double[,])_weights.Clone();
                var numNeurons = trainingData[0].Length;
                for (int i = 0; i < numNeurons; i++)
                {
                    for (int j = 0; j < numNeurons; j++)
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
                Console.WriteLine(diffNorm(old, _weights));

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

        public int[] Predict(int[] _input)
        {
            var iterations = 20;
            var input = (int[])_input.Clone();
            var output = new int[input.Length];
            while(iterations > 0)
            {
                //Synch
                for(int i = 0; i < output.Length; i++)
                {
                    var sum = 0.0;
                    for (int j = 0; j < output.Length; j++)
                    {
                        sum += _weights[i, j] * input[j];
                    }

                    if (sum > 0)
                    {
                        output[i] = 1;
                    } else if (sum < 0)
                    {
                        output[i] = -1;
                    }
                    else
                    {
                        output[i] = 0;
                    }
                }

                for (int i = 0; i < input.Length; i++)
                {
                    input[i] = output[i];
                }

                iterations--;
            }

            return input;
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
