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
        private double[,] _weights;
        public HopfieldNetwork(int neuronCount)
        {
            _neuronCount = neuronCount;
            _weights = new double[neuronCount, neuronCount];
        }

        public void Train(double[][] trainingData)
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

        public void Predict(double[] input)
        {
            var iterations = 20;
            var output = new double[input.Length];
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
        }
    }
}
