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

# load variables
#=
mf = matopen("/home/naxos2-raid1/maxiao/Projects/pTXDL/JuliaPrograms/ImageBasedDLForpTX.jl/Data/Data_processed.mat")
Xtrain = read(mf, "Xtrain")
Ytrain = read(mf, "Ytrain")
close(mf)

Xtrain = Xtrain./maximum(Xtrain)
Ytrain = Ytrain./maximum(Ytrain)

Xall = Xtrain
Yall = Ytrain

Xtrain = Xall[:,:,:,:,1:4]
Ytrain = Yall[:,:,:,:,1:4]

Sall = size(Xtrain)
Xtrain = reshape(Xtrain,Sall[1],Sall[2],1,Sall[3]*Sall[4]*Sall[5])

Sall = size(Ytrain)
Ytrain = reshape(Ytrain,Sall[1],Sall[2],1,Sall[3]*Sall[4]*Sall[5])

XValid = Xall[:,:,:,:,5] 
YValid = Yall[:,:,:,:,5]

Xall = nothing;
Yall = nothing;



mf = matopen("/home/naxos2-raid1/maxiao/Projects/pTXDL/JuliaPrograms/ImageBasedDLForpTX.jl/Data/T1wBrain_processed.mat")
T1wBrain_resize_ds = read(mf, "T1wBrain_resize_ds")
close(mf)

T1wBrain_resize_ds = T1wBrain_resize_ds./maximum(T1wBrain_resize_ds)

Xtrain_new = zeros(Float32,size(Xtrain,1),size(Xtrain,1),2,size(Xtrain,4))
Xtrain_new[:,:,1,:] = Xtrain
Xtrain_new[:,:,2,:] = reshape(repeat(T1wBrain_resize_ds[:,:,:,1:4],1,1,7,1),size(Xtrain,1),size(Xtrain,1),1,size(Xtrain,4))

Xtrain = Xtrain_new
Xtrain_new=nothing

Sall = size(XValid)
XValid = reshape(XValid,Sall[1],Sall[2],1,Sall[3]*Sall[4])
YValid = reshape(YValid,Sall[1],Sall[2],1,Sall[3]*Sall[4])

XValid_new = zeros(Float32,size(XValid,1),size(XValid,1),2,size(XValid,4))
XValid_new[:,:,1,:] = XValid
XValid_new[:,:,2,:] = reshape(repeat(T1wBrain_resize_ds[:,:,:,5],1,1,7),size(XValid,1),size(XValid,1),1,size(XValid,4))

XValid = XValid_new
XValid_new=nothing

JLD.save("Data/XYtrain_XYvalid_processed_T1w.jld","Xtrain",Xtrain,"Ytrain",Ytrain,"XValid",XValid,"YValid",YValid)

=#


Xtrain = JLD.load("Data/XYtrain_XYvalid_processed_T1w.jld","Xtrain")
Ytrain = JLD.load("Data/XYtrain_XYvalid_processed_T1w.jld","Ytrain")
XValid = JLD.load("Data/XYtrain_XYvalid_processed_T1w.jld","XValid")
YValid = JLD.load("Data/XYtrain_XYvalid_processed_T1w.jld","YValid")

#=
fig, ax = plt.subplots()
PyPlot.gray()
ax.imshow(abs.(Xtrain[:,:,1,1]))
fig, ax = plt.subplots()
PyPlot.gray()
ax.imshow(abs.(Xtrain[:,:,2,1]))
fig2, ax2 = plt.subplots()
PyPlot.gray()
ax2.imshow(abs.(Ytrain[:,:,1,1]))
#display(fig)
=#


idx_shuffle = shuffle(collect(1:size(Xtrain,4)))
Xtrain = Xtrain[:,:,:,idx_shuffle]
Ytrain = Ytrain[:,:,:,idx_shuffle]

Xtrain = gpu(Xtrain)
Ytrain = gpu(Ytrain)

batch_size = 10

train_loader = Flux.Data.DataLoader(Xtrain,Ytrain, batchsize=batch_size,shuffle=true)

XValid = gpu(XValid)
YValid = gpu(YValid)


channels_in=2
channels_out=1

# Unet define
function _random_normal(shape...)
  return Float32.(rand(Normal(0.f0,0.02f0),shape...))
end

expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)
function squeeze(x) 
    if size(x)[end] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    else
        # For the case BATCH_SIZE = 1
        int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)...,1)
    end
end

BatchNormWrap(out_ch) = BatchNorm(out_ch)

UNetConvBlock(in_chs, out_chs, kernel = (3, 3)) =
    Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1);init=_random_normal),
	BatchNormWrap(out_chs),
	x->leakyrelu.(x,0.2f0))

ConvDown(in_chs,out_chs,kernel = (4,4)) =
  Chain(Conv(kernel,in_chs=>out_chs,pad=(1,1),stride=(2,2);init=_random_normal),
	BatchNormWrap(out_chs),
	x->leakyrelu.(x,0.2f0))

UNetUpBlock(in_chs, out_chs; kernel = (3, 3), p = 0.5f0) = 
  Chain(x->leakyrelu.(x,0.2f0),
  	ConvTranspose((2, 2), in_chs=>out_chs,stride=(2, 2);init=_random_normal),
	BatchNormWrap(out_chs),
	Dropout(p))

u = Chain(
       UNetConvBlock(channels_in, 64), 
       UNetConvBlock(64, 64), #op
       SkipConnection(Chain(ConvDown(64,64),
                            UNetConvBlock(64, 128),
                            UNetConvBlock(128, 128),#x1
                            SkipConnection(Chain(ConvDown(128,128),
		                                 UNetConvBlock(128, 256),
		                                 UNetConvBlock(256, 256),#x2
		                                 SkipConnection(Chain(ConvDown(256,256),
				                                      UNetConvBlock(256, 512),
				                                      UNetConvBlock(512, 512),#x3
				                                      SkipConnection(Chain(ConvDown(512,512),
					                                                   UNetConvBlock(512, 1024),#x4
					                                                   UNetConvBlock(1024, 1024),#up_x4
					                                                   UNetUpBlock(1024,512)),
					                                                   (mx,x)->cat(mx,x,dims=3)),
				                                      UNetConvBlock(1024, 512),
				                                      UNetConvBlock(512, 512),#up_x1
				                                      UNetUpBlock(512,256)),
				                                      (mx,x)->cat(mx,x,dims=3)),
                                                 UNetConvBlock(512, 256),
                                                 UNetConvBlock(256, 256),#up_x2
                                                 UNetUpBlock(256,128)),
                                                 (mx,x)->cat(mx,x,dims=3)),
                            UNetConvBlock(256, 128),
                            UNetConvBlock(128, 128),#up_x3
                            UNetUpBlock(128,64)),
                            (mx,x)->cat(mx,x,dims=3)),
       UNetConvBlock(128, 64),
       UNetConvBlock(64, 64),#up_x5
       Chain(x->leakyrelu.(x,0.2f0),Conv((1, 1), 64=>channels_out;init=_random_normal))
)




u = gpu(u)

loss(x,y) = Flux.mse(u(x),y)
#loss(x,y) = mean((u(x).-y).^2)
validationloss() = mean(Flux.mse(u(XValid[:,:,:,(idx-1)*100+1:idx*100]),YValid[:,:,:,(idx-1)*100+1:idx*100]) for idx=1:round(Int,size(XValid,4)/100))
#validationloss() = mean(Flux.mse(u(XValid[:,:,:,idx]),YValid[:,:,:,idx]) for idx=1:size(XValid,4))

#CuArrays.@cufunc loss_cu(x,y) = mean((u(x).-y).^2)

#=
θ = params(u) 
gs = Flux.gradient(θ) do
  training_loss = loss(XValid[:,:,:,1:2],YValid[:,:,:,1:2])
end
#=

#cb = Flux.throttle(() -> @show(validationloss()), 10)
#@time Flux.train!(loss, θ, dataset, opt, cb = cb)
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
  
loss_valid = zeros(10)
for epoch in 1:10
  #@time Flux.train!(loss, Flux.params(u), train_loader, opt, cb = cb)
  @time my_custom_train!(loss, Flux.params(u), train_loader, opt)
  #@time Flux.train!(loss, Flux.params(u), train_set, opt)
  model = cpu(u)
  BSON.@save string("Results/model_unet_ptx_processed_7dir_T1w_gpu_epoch",string(epoch),".bson") model
  u = gpu(model)
  loss_valid[epoch] = validationloss()
end



#test
#=
XValid = JLD.load("XYtrain_XYvalid_processed.jld","XValid")
YValid = JLD.load("XYtrain_XYvalid_processed.jld","YValid")
=#
Xtest = zeros(Float32,size(XValid,1),size(XValid,2),2,1)
Ytest = zeros(Float32,size(XValid,1),size(XValid,2),1,1)

idx_dir_test = 2
idx_slc_test = 15
idx_test = idx_slc_test + (idx_dir_test-1)*100
Xtest[:,:,:,1] = XValid[:,:,:,idx_test]
Ytest[:,:,:,1] = YValid[:,:,:,idx_test]
Xtest_output = u(gpu(Xtest))

fig, ax = plt.subplots()
PyPlot.gray()
ax.imshow(abs.(Xtest[:,:,1,1]))

fig2, ax2 = plt.subplots()
PyPlot.gray()
#savefig("Xttrain_40.png",fig)
ax2.imshow(abs.(Xtest_output[:,:,1,1]))

fig3, ax3 = plt.subplots()
PyPlot.gray()
ax3.imshow(abs.(Ytest[:,:,1,1]))


BSON.@load string("Results/model_unet_ptx_processed_7dir_T1w_gpu_epoch",string(10),".bson") model
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

slc_list = [10 25]
dir_list = [1 2]
XtestOutput4write_slc10_25_dir1_2 = zeros(Float32,size(XValid,1),size(XValid,2),length(slc_list),length(dir_list),10)
for idx_epoch in 1:10
  BSON.@load string("Results/model_unet_ptx_processed_7dir_T1w_gpu_epoch",string(idx_epoch),".bson") model
  Xtest = zeros(Float32,size(XValid,1),size(XValid,2),2,1)
  Ytest = zeros(Float32,size(XValid,1),size(XValid,2),1,1)
  for idx_dir_test in 1:length(dir_list)
    for idx_slc_test in 1:length(slc_list)
      idx_test = idx_slc_test + (idx_dir_test-1)*100
      Xtest[:,:,:,1] = XValid[:,:,:,idx_test]
      XtestOutput4write_slc10_25_dir1_2[:,:,idx_slc_test,idx_dir_test,idx_epoch] = model(Xtest)
    end
  end
end


file = matopen("Results/XValidOutput_subj5_processed_7dir_T1w_gpu_epoch10.mat", "w")
write(file, "XtestOutput4write", XtestOutput4write)
write(file, "XtestOutput4write_slc10_25_dir1_2", XtestOutput4write_slc10_25_dir1_2)
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
