using Flux
using Zygote
using MAT
#using Plots
using Statistics
using Distributions
using LinearAlgebra
using Random
using BSON
using JLD
using PyPlot
using CUDA
#using Base.Iterators: partition

gpu_id = 1
CUDA.device!(gpu_id)

Xtrain = JLD.load("Data/XYtrain_XYvalid_processed_T1w.jld","Xtrain")
Ytrain = JLD.load("Data/XYtrain_XYvalid_processed_T1w.jld","Ytrain")
XValid = JLD.load("Data/XYtrain_XYvalid_processed_T1w.jld","XValid")
YValid = JLD.load("Data/XYtrain_XYvalid_processed_T1w.jld","YValid")

idx_shuffle = shuffle(collect(1:size(Xtrain,4)))
Xtrain = Xtrain[:,:,:,idx_shuffle]
Ytrain = Ytrain[:,:,:,idx_shuffle]

Xtrain = reshape(Xtrain,size(Xtrain,1),size(Xtrain,1),2,size(Xtrain,4))

Xtrain = gpu(Xtrain)
Ytrain = gpu(Ytrain)

batch_size = 10

train_loader = Flux.Data.DataLoader(Xtrain,Ytrain, batchsize=batch_size,shuffle=true)

#XValid = XValid[:,:,1,:]
XValid = reshape(XValid,size(XValid,1),size(XValid,1),2,size(XValid,4))

XValid = gpu(XValid)
YValid = gpu(YValid)


channels_in=2
channels_out=1

# Unet define
function _random_normal(shape...)
  return Float32.(rand(Normal(0.f0,0.02f0),shape...))
end

BatchNormWrap(out_ch) = BatchNorm(out_ch)

ResNetConvBlock(chs, kernel = (3, 3)) =
    Chain(Conv(kernel, chs=>chs,pad = (1, 1);init=_random_normal),
          BatchNorm(chs),
          x->relu.(x),
          Conv(kernel, chs=>chs,pad = (1, 1);init=_random_normal),
          BatchNorm(chs))

r = Chain(
            Conv((4,4),channels_in=>64,pad=(1,1),stride=(2,2);init=_random_normal),
            x->relu.(x),
            SkipConnection(ResNetConvBlock(64),(mx,x)->mx+x),
            SkipConnection(ResNetConvBlock(64),(mx,x)->mx+x),
            SkipConnection(ResNetConvBlock(64),(mx,x)->mx+x),
            SkipConnection(ResNetConvBlock(64),(mx,x)->mx+x),
            ConvTranspose((2,2),64=>channels_out,stride=(2,2);init=_random_normal),
            BatchNorm(channels_out)
        )

r = gpu(r)

#loss(x,y) = mean(abs,(u(x).-y))
loss(x,y) = mean((r(x).-y).^2)
validationloss() = mean(Flux.mse(r(XValid[:,:,:,(idx-1)*100+1:idx*100]),YValid[:,:,:,(idx-1)*100+1:idx*100]) for idx=1:round(Int,size(XValid,4)/100))
#validationloss() = mean(mean(abs,(u(XValid[:,:,:,(idx-1)*100+1:idx*100]).-YValid[:,:,:,(idx-1)*100+1:idx*100])) for idx=1:round(Int,size(XValid,4)/100))

opt = ADAM(0.001)


function my_custom_train!(loss, ps, data, opt)
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss
    ps = Params(ps)
    for d in data
      if d isa AbstractArray{<:Number}
        gs = Flux.gradient(ps) do
          training_loss = loss(d)
          return training_loss
        end
      else
        gs = Flux.gradient(ps) do
          training_loss = loss(d...)
          return training_loss
        end
      end
      # Insert whatever code you want here that needs training_loss, e.g. logging.
      # logging_callback(training_loss)
      # Insert what ever code you want here that needs gradient.
      # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
      #println("training_loss = ",training_loss,"; gradient[1] = ",gs[ps[1]],"; W1 = ",ps.order.data[1][1],"; b1 = ",ps.order.data[2][1])
      println("training_loss = ",training_loss)
      Flux.update!(opt, ps, gs)
      # Here you might like to check validation set accuracy, and break out to do early stopping.
    end
  end

loss_valid = zeros(100)
for epoch in 1:100
  learn_rate = 0.001*exp(log(0.1)/99*(epoch-1))
  opt = ADAM(learn_rate)
  #@time Flux.train!(loss, Flux.params(u), train_loader, opt, cb = cb)
  @time my_custom_train!(loss, Flux.params(r), train_loader, opt)
  #@time Flux.train!(loss, Flux.params(u), train_set, opt)
  #model = cpu(u)
  #BSON.@save string("Results/model_resnet_ptx_processed_7dir_gpu_epoch",string(epoch),".bson") model
  #u = gpu(model)
  loss_valid[epoch] = validationloss()
end

model = cpu(r)
BSON.@save string("Results/model_resnet_ptx_processed_7Vol_gpu_T1w_epoch100.bson") model

#=
loss_valid = zeros(50)
for epoch in 1:50
  learn_rate = 0.001*exp(log(0.1)/99*(epoch-1))
  opt = ADAM(learn_rate)
  @time Flux.train!(loss, Flux.params(r), train_loader, opt)
  #@time my_custom_train!(loss, Flux.params(r), train_loader, opt)
  #@time Flux.train!(loss, Flux.params(u), train_set, opt)
  #model = cpu(u)
  #BSON.@save string("Results/model_resnet_ptx_processed_7dir_gpu_epoch",string(epoch),".bson") model
  #u = gpu(model)
  loss_valid[epoch] = validationloss()
end

model = cpu(r)
BSON.@save string("Results/model_resnet_ptx_processed_7Vol_gpu_T1w_epoch50.bson") model
=#

#test
#=
XValid = JLD.load("XYtrain_XYvalid_processed.jld","XValid")
YValid = JLD.load("XYtrain_XYvalid_processed.jld","YValid")
=#


BSON.@load string("Results/model_resnet_ptx_processed_7Vol_gpu_T1w_epoch50.bson") model
slc_list = collect(1:100)
dir_list = collect(1:7)
XtestOutput4write = zeros(size(XValid,1),size(XValid,2),length(slc_list),length(dir_list))

Xtest = zeros(Float32,size(XValid,1),size(XValid,2),2,1)
Ytest = zeros(Float32,size(XValid,1),size(XValid,2),1,1)
for idx_dir_test in 1:length(dir_list)
  for idx_slc_test in 1:length(slc_list)
    idx_test = idx_slc_test + (idx_dir_test-1)*100
    Xtest[:,:,:,1] = XValid[:,:,:,idx_test]
    #Ytest[:,:,1,1] = YValid[:,:,idx_slc_test,idx_dir_test]
    XtestOutput4write[:,:,idx_slc_test,idx_dir_test] = model(Xtest)
  end
end


file = matopen("Results/XValidOutput_subj5_processed_7Vol_gpu_resnet_T1w_epoch50.mat", "w")
write(file, "XtestOutput4write", XtestOutput4write)
close(file)








# testing on training data
BSON.@load string("model_unet_ptx_processed_7dir_epoch",string(10),".bson") u
XValid = Xtrain[:,:,:,1:700]
XValid = reshape(XValid,size(XValid,1),size(XValid,2),100,7)
XtestOutput4write = zeros(size(XValid))

Xtest = zeros(Float32,size(XValid,1),size(XValid,2),1,1)
Ytest = zeros(Float32,size(XValid,1),size(XValid,2),1,1)
for idx_dir_test in 1:size(XValid,4)
  for idx_slc_test in 1:size(XValid,3)
    Xtest[:,:,1,1] = XValid[:,:,idx_slc_test,idx_dir_test]
    #Ytest[:,:,1,1] = YValid[:,:,idx_slc_test,idx_dir_test]
    XtestOutput4write[:,:,idx_slc_test,idx_dir_test] = u(Xtest)
  end
end

file = matopen("XValidOutput_traingsubj1_processed_7dir_epoch10.mat", "w")
write(file, "XtestOutput4write", XtestOutput4write)
close(file)
