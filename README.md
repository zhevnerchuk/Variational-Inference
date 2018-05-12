# Variational-Inference
Project for LSML course at YSDA

If you want to intsall via Docker, download folder `docker`, when build Docker Image from it via terminal:

    sudo docker build -t zhmar -f Dockerfile .
    
Then execute the following command to run it:

    sudo docker run -p 9999:8888 -it zhmar:latest

You can find documenatation for this library in file ![ZhMaR.pdf](https://github.com/zhevnerchuk/Variational-Inference/blob/master/ZhMAR_docs.pdf)

**Team:**
  * Sergey Makarychev
  * Aleksandr Rozhnov
  * Anton Zhevnerchuk
  

**Постановка задачи:**

Имеется вероятностная модель ![p(x,z)](https://latex.codecogs.com/gif.latex?p_%7B%5Ctheta%7D%28x%2C%20z%29), где x -- вектор наблюдаемых, z -- скрытых переменных, а ![theta](https://latex.codecogs.com/gif.latex?%24%5Ctheta%24) -- вектор параметров.

Цель: по набору наблюдаемых x-в восстановить оптимальное ![theta](https://latex.codecogs.com/gif.latex?%24%5Ctheta%24):

![as](https://latex.codecogs.com/gif.latex?%5Clog%20p_%7B%5Ctheta%7D%28x%29%20%5Cto%20%5Cmax.)

Для решения этой задачи вводится вариационное распределение ![q(z)](https://latex.codecogs.com/gif.latex?q_%7B%5Cpsi%7D%28z%29), которым аппроксимируется постериорное распределение на z:

![apprx](https://latex.codecogs.com/gif.latex?q_%7B%5Cpsi%7D%28z%29%20%5Capprox%20p%28z%20%7C%20X%29.)

Максимизировать непосредственно правдоподобие сложно: в формулах возникают интегралы, которые в большинстве случаев не берутся аналитически. Кроме того, зачастую такой подход плохо масштабируется на большие объемы выборок. По этой причине вместо правдоподобия максимизируют ELBO (Evidence Lower BOund).

![setting](https://latex.codecogs.com/gif.latex?%5Clog%20p_%7B%5Ctheta%7D%28x%29%20%5Cgeq%20%5Cmathrm%7BELBO%7D%20%3A%3D%20%5Cmathbb%7BE%7D_%7Bz%20%5Csim%20q_%7B%5Cpsi%7D%28z%29%7D%20%5B%20%5Clog%20p_%7B%5Ctheta%7D%28x%2C%20z%29%20-%20%5Clog%20q_%7B%5Cpsi%7D%28z%29%20%5D)

Оптимизировать ELBO можно итерируясь по небольшим батчам, поэтому в такой постановке задача масштабируется на случай больших данных. Есть несколько способов считать ELBO (подробнее про это можно почитать, например, [тут](http://andymiller.github.io/2016/12/19/elbo-gradient-estimators.html)). 

Существующие библиотеки для Вариационного Вывода:
  * [Edward](http://edwardlib.org/) (работает с TensorFlow и Keras)
  * [Pyro](http://pyro.ai/) (работает с PyTorch)

На данный момент реализован самый наивный (и одновременно наиболее общий) способ аппроксимации ELBO -- методом Монте-Карло. Реализация написана на питоне и использует PyTorch.

Дальнейшие шаги:
  * Реализовать другие способы подсчета ELBO (Importance Sampling, с использованием KL-дивергенции и энтропии для тех классов распределений, для которых они считаются аналитически)
  * Переписать все на C++ (хотелось бы в итоге иметь совместимость с PyTorch и не реализовывать автодифференцирование самостоятельно, поэтому предполагается использовать [ATen](https://github.com/pytorch/pytorch/tree/master/aten)). Использование PyTorch позволит работать с динамическими графами вычислений (чего нет в Edward). Хотелось бы по-крайней мере для каких-то моделей получить ускорение относительно Pyro.
  * Попробовать делать вариационный вывод с другими функциями потерь (не ELBO). На данный момент этого нет в Pyro. Один из возможных вариантов альтернативной функции потерь изложен [тут](https://papers.nips.cc/paper/6866-variational-inference-via-chi-upper-bound-minimization.pdf)).
  
