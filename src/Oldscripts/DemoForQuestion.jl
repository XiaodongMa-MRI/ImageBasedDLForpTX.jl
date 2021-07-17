using Flux
using Zygote
using CUDA

X = rand(Float32,128,128,2,10)
Y = rand(Float32,128,128,1,10)

model = Conv((3,3),2=>1,pad=(1,1),relu)

X = gpu(X)
Y = gpu(Y)
model = gpu(model)

loss(x,y) = Flux.mse(model(x),y)

train_loader = Flux.Data.DataLoader(X,Y, batchsize=2,shuffle=true)

opt = ADAM(0.001)
@time Flux.train!(loss, Flux.params(model), train_loader, opt)




