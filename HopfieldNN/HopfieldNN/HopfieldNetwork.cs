using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HopfieldNN
{
    public class HopfieldNetwork
    {
        private int _neuronCount;
        private double _lr;
        private string _rule;
        private double[,] _weights;
        private Random _rng;
        public HopfieldNetwork(int neuronCount, string rule, double lr=0.001)
        {
            _neuronCount = neuronCount;
            _lr = lr;
            _rule = rule;
            _weights = new double[neuronCount, neuronCount];
            _rng = new Random(42);
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
                    throw new Exception();
            }
        }

        private void trainOja(int[][] trainingData)
        {
            for (int i = 0; i < _weights.GetLength(0); i++)
            {
                for (int j = 0; j < _weights.GetLength(1); j++)
                {
                    _weights[i, j] = (_rng.NextDouble() * 2.0 - 1.0);// * 0.01;
                }
            }
            for (int q = 0; q < 100; q++)
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

        private int[] dot(int[] a, int[] b)
        {
            return a.Zip(b, (a,b) => a*b).ToArray();
        }

        public int[] Predict(int[] input)
        {
            var iterations = 20;
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
    }
}
