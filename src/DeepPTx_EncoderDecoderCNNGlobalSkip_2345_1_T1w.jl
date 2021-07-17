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
#using PyPlot
using CUDA
#using Base.Iterators: partition

#gpu_id = 1
#CUDA.device!(gpu_id)


#****************** load data ******************
mf = matopen("../../Data/Data_processed_36Vol_NoCrop.mat")
Xall = read(mf, "Xtrain");
Yall = read(mf, "Ytrain");
close(mf)

for idx_subj = 1:size(Xall,5)
    Xall[:,:,:,:,idx_subj] = Xall[:,:,:,:,idx_subj]./maximum(Xall[:,:,:,:,idx_subj]);
    Yall[:,:,:,:,idx_subj] = Yall[:,:,:,:,idx_subj]./maximum(Yall[:,:,:,:,idx_subj]);
end
Xall = Xall[:,:,37:136,:,:];
Yall = Yall[:,:,37:136,:,:];

Sall = size(Xall);
Xall = reshape(Xall,Sall[1],Sall[2],1,Sall[3]*Sall[4],Sall[5]);

Sall = size(Yall);
Yall = reshape(Yall,Sall[1],Sall[2],1,Sall[3]*Sall[4],Sall[5]);
NVol = Sall[4] # number of volumes, 36

mf = matopen("../../Data/T1wBrain_processed_NoCrop.mat")
T1wBrain_resize_ds = read(mf, "T1wBrain_resize_ds");
close(mf)

for idx_subj = 1:size(T1wBrain_resize_ds,4)
    T1wBrain_resize_ds[:,:,:,idx_subj] = T1wBrain_resize_ds[:,:,:,idx_subj]./maximum(T1wBrain_resize_ds[:,:,:,idx_subj]);
end
T1wBrain_resize_ds = T1wBrain_resize_ds[:,:,37:136,:];

Xall_new = zeros(Float32,size(Xall,1),size(Xall,2),2,size(Xall,4),size(Xall,5));
Xall_new[:,:,1,:,:] = Xall;
Xall_new[:,:,2,:,:] = reshape(repeat(T1wBrain_resize_ds,1,1,NVol,1),size(Xall,1),size(Xall,2),1,size(Xall,4),size(Xall,5));

Xall = Xall_new;
Xall_new=nothing;

Sall = size(Xall);
Xtrain = reshape(Xall[:,:,:,:,[2,3,4,5]],Sall[1],Sall[2],Sall[3],Sall[4]*4);
Ytrain = reshape(Yall[:,:,:,:,[2,3,4,5]],Sall[1],Sall[2],1,Sall[4]*4);
idx_shuffle = shuffle(collect(1:size(Xtrain,4)));
Xtrain = Xtrain[:,:,:,idx_shuffle];
Ytrain = Ytrain[:,:,:,idx_shuffle];

# for noT1w only
#=
Xtrain_new = zeros(size(Ytrain));
Xtrain_new[:,:,1,:] = Xtrain[:,:,1,:];
Xtrain = Xtrain_new;
Xtrain_new = nothing;
=#

#Ytrain[:,:,1,:] = (Ytrain[:,:,1,:]-Xtrain[:,:,1,:])./(2.0).+(0.5)

Nim = size(Xtrain,1);

XValid = reshape(Xall[:,:,:,:,1],Sall[1],Sall[2],Sall[3],Sall[4]);
YValid = reshape(Yall[:,:,:,:,1],Sall[1],Sall[2],1,Sall[4]);

Xall = nothing;
Yall = nothing;

# for noT1w only
#=
XValid_new = zeros(size(YValid));
XValid_new[:,:,1,:] = XValid[:,:,1,:];
XValid = XValid_new;
XValid_new = nothing;
=#

# put data on GPU
Xtrain = gpu(Xtrain);
Ytrain = gpu(Ytrain);
XValid = gpu(XValid);
YValid = gpu(YValid);


#****************** Hyperparameters ******************
channels_in=size(Xtrain,3);
channels_out=1;

Nch_init = 16;
kernelsize = (3, 3);

batch_size = 32;
Nepochs = 30;
LR_initial = 10e-4;
LR_decay_factor = 0.15;
LR_decay_step = 8;
#Nblocks = nb_list[idx_randsearch]

println("  batch_size      = ",batch_size)
println("  Nepochs         = ",Nepochs)
println("  LR_initial      = ",LR_initial)
println("  LR_decay_factor = ",LR_decay_factor)
println("  LR_decay_step   = ",LR_decay_step)
#println("  Nblocks         = ",Nblocks)


#****************** neural network ******************
BatchNormWrap(out_ch) = BatchNorm(out_ch)

UNetConvBlock(in_chs, out_chs, kernel = kernelsize) =
    Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1);init=Flux.kaiming_normal),
	BatchNormWrap(out_chs),
	x->relu.(x))

#ConvDown(in_chs,out_chs,kernel = (2,2)) = Conv(kernel,in_chs=>out_chs,stride=(2,2);init=Flux.kaiming_normal)

UNetUpBlock(in_chs, out_chs; kernel = (2,2)) = ConvTranspose(kernel, in_chs=>out_chs,stride=(2, 2);init=Flux.kaiming_normal)

r = SkipConnection(Chain(
       UNetConvBlock(channels_in, Nch_init), 
       UNetConvBlock(Nch_init, Nch_init), #op
       SkipConnection(Chain(MaxPool((2,2)),
                            UNetConvBlock(Nch_init, Nch_init*2),
                            UNetConvBlock(Nch_init*2, Nch_init*2),#x1
                            SkipConnection(Chain(MaxPool((2,2)),
		                                 UNetConvBlock(Nch_init*2, Nch_init*4),
		                                 UNetConvBlock(Nch_init*4, Nch_init*4),#x2
		                                 SkipConnection(Chain(MaxPool((2,2)),
				                                      UNetConvBlock(Nch_init*4, Nch_init*8),
				                                      UNetConvBlock(Nch_init*8, Nch_init*8),#x3
				                                      UNetUpBlock(Nch_init*8,Nch_init*4)),
				                                (mx,x)->cat(mx,x,dims=3)),
                                                 UNetConvBlock(Nch_init*8, Nch_init*4),
                                                 UNetConvBlock(Nch_init*4, Nch_init*4),#up_x2
                                                 UNetUpBlock(Nch_init*4,Nch_init*2)),
                                           (mx,x)->cat(mx,x,dims=3)),
                            UNetConvBlock(Nch_init*4, Nch_init*2),
                            UNetConvBlock(Nch_init*2, Nch_init*2),#up_x3
                            UNetUpBlock(Nch_init*2,Nch_init)),
                      (mx,x)->cat(mx,x,dims=3)),
       UNetConvBlock(Nch_init*2, Nch_init),
       UNetConvBlock(Nch_init, Nch_init),#up_x5
       Conv((1, 1), Nch_init=>channels_out;init=Flux.kaiming_normal)),
       (mx,x)->reshape(mx[:,:,1,:]+x[:,:,1,:],Nim,Nim,1,size(x,4)))

r = gpu(r)

#model to save
model = SkipConnection(Chain(
       UNetConvBlock(channels_in, Nch_init), 
       UNetConvBlock(Nch_init, Nch_init), #op
       SkipConnection(Chain(MaxPool((2,2)),
                            UNetConvBlock(Nch_init, Nch_init*2),
                            UNetConvBlock(Nch_init*2, Nch_init*2),#x1
                            SkipConnection(Chain(MaxPool((2,2)),
		                                 UNetConvBlock(Nch_init*2, Nch_init*4),
		                                 UNetConvBlock(Nch_init*4, Nch_init*4),#x2
		                                 SkipConnection(Chain(MaxPool((2,2)),
				                                      UNetConvBlock(Nch_init*4, Nch_init*8),
				                                      UNetConvBlock(Nch_init*8, Nch_init*8),#x3
				                                      UNetUpBlock(Nch_init*8,Nch_init*4)),
				                                (mx,x)->cat(mx,x,dims=3)),
                                                 UNetConvBlock(Nch_init*8, Nch_init*4),
                                                 UNetConvBlock(Nch_init*4, Nch_init*4),#up_x2
                                                 UNetUpBlock(Nch_init*4,Nch_init*2)),
                                           (mx,x)->cat(mx,x,dims=3)),
                            UNetConvBlock(Nch_init*4, Nch_init*2),
                            UNetConvBlock(Nch_init*2, Nch_init*2),#up_x3
                            UNetUpBlock(Nch_init*2,Nch_init)),
                      (mx,x)->cat(mx,x,dims=3)),
       UNetConvBlock(Nch_init*2, Nch_init),
       UNetConvBlock(Nch_init, Nch_init),#up_x5
       Conv((1, 1), Nch_init=>channels_out;init=Flux.kaiming_normal)),
       (mx,x)->reshape(mx[:,:,1,:]+x[:,:,1,:],Nim,Nim,1,size(x,4)))


#****************** Data loader and Train/Loss function ******************
train_loader = Flux.Data.DataLoader(Xtrain,Ytrain, batchsize=batch_size,shuffle=true)
loss(x,y) = Flux.mse(r(x),y)
validationloss() = mean(Flux.mse(r(XValid[:,:,:,(idx-1)*100+1:idx*100]),YValid[:,:,:,(idx-1)*100+1:idx*100]) for idx=1:round(Int,size(XValid,4)/100))
trainingloss() = mean(Flux.mse(r(Xtrain[:,:,:,(idx-1)*100+1:idx*100]),Ytrain[:,:,:,(idx-1)*100+1:idx*100]) for idx=1:round(Int,size(Xtrain,4)/100))
#trainingloss() = Flux.mse(r(Xtrain),Ytrain)
#validationloss() = Flux.mse(r(XValid),YValid)


Savename = "EncoderDecoderGlobalSkip_2345_1_T1W";

#****************** Training ******************
loss_train = zeros(Float32,Nepochs);
loss_valid = zeros(Float32,Nepochs);
NepochStop = 0;
for epoch in 1:Nepochs
  learn_rate = LR_initial*exp(-LR_decay_factor*floor((epoch-1)./LR_decay_step))
  opt = ADAM(learn_rate)
  @time Flux.train!(loss, Flux.params(r), train_loader, opt)
  #BSON.@save string("Results/model_resnet_ptx_processed_7dir_gpu_epoch",string(epoch),".bson") model
  #CUDA.unsafe_free!(u)
  #u = gpu(uc)
  loss_train[epoch] = trainingloss()
  loss_valid[epoch] = validationloss()
  
  println("Training epoch No.",epoch,", loss_train = ", loss_train[epoch])
  println("Training epoch No.",epoch,", loss_valid = ", loss_valid[epoch])
  
#=  if (epoch>10) && (loss_valid[epoch]>loss_valid[epoch-1])
    NepochStop = epoch;
    break;
  end=#
  model = cpu(r);
  BSON.@save string("Results/model_",Savename,"Epoch",epoch,".bson") model

end

valmin, indmin = findmin(loss_valid);
for epoch in 1:Nepochs
  if epoch != indmin
    rm(string("Results/model_",Savename,"Epoch",epoch,".bson"));
  end
end

#****************** Save ******************
#model = cpu(r);
JLD.save(string("Results/ParsResults_",Savename,".jld"),
         "loss_train",loss_train,
         "loss_valid",loss_valid,
         "NepochStop",indmin,
         "batch_size",batch_size,
         "LR_initial",LR_initial,
         "LR_decay_factor",LR_decay_factor,
         "LR_decay_step",LR_decay_step)

