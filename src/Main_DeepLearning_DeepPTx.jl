using Flux
using Zygote
using MAT
using Statistics
using Distributions
using LinearAlgebra
using Random
using BSON
using JLD
using CUDA

#****************** load data ******************
mf = matopen("../Data/Data_preprocessed_7vol.mat")
Xtrain = read(mf, "Xtrain"); #T1w + sTx DW images: Nx*Ny*2(T1w+DWI)*Nslice*Nvolume*Nsubjects
Ytrain = read(mf, "Ytrain"); #pTx DW images: Nx*Ny*1*Nslice*Nvolume*Nsubjects
Xvalid = read(mf, "Xvalid"); #T1w + sTx DW images: Nx*Ny*2(T1w+DWI)*Nslice*Nvolume*Nsubjects
Yvalid = read(mf, "Yvalid"); #pTx DW images: Nx*Ny*1*Nslice*Nvolume*Nsubjects
close(mf)

Sall = size(Xtrain);
Xtrain = reshape(Xtrain,Sall[1],Sall[2],Sall[3],Sall[4]*Sall[5]*Sall[6]);
Ytrain = reshape(Ytrain,Sall[1],Sall[2],1,Sall[4]*Sall[5]*Sall[6]);
Xvalid = reshape(Xvalid,Sall[1],Sall[2],Sall[3],Sall[4]*Sall[5]*1);
Yvalid = reshape(Yvalid,Sall[1],Sall[2],1,Sall[4]*Sall[5]*1);

Nim = size(Xtrain,1);

if CUDA.functional()
    Xtrain = gpu(Xtrain);
    Ytrain = gpu(Ytrain);
    Xvalid = gpu(Xvalid);
    Yvalid = gpu(Yvalid);
end


# Unet define
#****************** neural network ******************

#Hyperparameters
Nch_init = 24;
batch_size = 24;
Nepochs = 20;
LR_initial = 5.87e-3;
LR_decay_factor = 0.28;
LR_decay_step = 4;
Npool = 4;

channels_in=size(Xtrain,3);
channels_out=1;
kernelsize = (3, 3);

BatchNormWrap(out_ch) = BatchNorm(out_ch)

UNetConvBlock(in_chs, out_chs, kernel = kernelsize) =
    Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1);init=Flux.kaiming_normal),
	BatchNormWrap(out_chs),
	x->relu.(x))

ConvDown(in_chs,out_chs,kernel = (2,2)) = Conv(kernel,in_chs=>out_chs,stride=(2,2);init=Flux.kaiming_normal)

UNetUpBlock(in_chs, out_chs; kernel = (2,2)) = ConvTranspose(kernel, in_chs=>out_chs,stride=(2, 2);init=Flux.kaiming_normal)

model = SkipConnection(Chain(
           UNetConvBlock(channels_in, Nch_init), 
           UNetConvBlock(Nch_init, Nch_init), #op
           SkipConnection(Chain(ConvDown(Nch_init,Nch_init),
                                UNetConvBlock(Nch_init, Nch_init*2),
                                UNetConvBlock(Nch_init*2, Nch_init*2),#x1
                                SkipConnection(Chain(ConvDown(Nch_init*2,Nch_init*2),
                                                     UNetConvBlock(Nch_init*2, Nch_init*4),
                                                     UNetConvBlock(Nch_init*4, Nch_init*4),#x2
                                                     SkipConnection(Chain(ConvDown(Nch_init*4,Nch_init*4),
                                                                          UNetConvBlock(Nch_init*4, Nch_init*8),
                                                                          UNetConvBlock(Nch_init*8, Nch_init*8),#x3
                                                                          SkipConnection(Chain(ConvDown(Nch_init*8,Nch_init*8),
                                                                                               UNetConvBlock(Nch_init*8, Nch_init*16),
                                                                                               UNetConvBlock(Nch_init*16, Nch_init*16),#x3
                                                                                               UNetUpBlock(Nch_init*16,Nch_init*8)),
                                                                                         (mx,x)->cat(mx,x,dims=3)),
                                                                          UNetConvBlock(Nch_init*16, Nch_init*8),
                                                                          UNetConvBlock(Nch_init*8, Nch_init*8),#up_x2
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
           (mx,x)->reshape(mx[:,:,1,:]+x[:,:,1,:],Nim,Nim,1,size(x,4)));

if CUDA.functional()
    model = gpu(model);
end


#****************** Data loader and Train/Loss function ******************
train_loader = Flux.Data.DataLoader(Xtrain,Ytrain, batchsize=batch_size,shuffle=true);
loss(x,y) = Flux.mse(model(x),y);
trainingloss() = mean(Flux.mse(model(Xtrain[:,:,:,(idx-1)*100+1:idx*100]),Ytrain[:,:,:,(idx-1)*100+1:idx*100]) for idx=1:round(Int,size(Xtrain,4)/100));
validationloss() = mean(Flux.mse(model(Xvalid[:,:,:,(idx-1)*100+1:idx*100]),Yvalid[:,:,:,(idx-1)*100+1:idx*100]) for idx=1:round(Int,size(Xvalid,4)/100));

#****************** Training ******************
loss_train = zeros(Float32,Nepochs);
loss_valid = zeros(Float32,Nepochs);
for epoch in 1:Nepochs
  learn_rate = LR_initial*exp(-LR_decay_factor*floor((epoch-1)./LR_decay_step))
  opt = ADAM(learn_rate)
  @time Flux.train!(loss, Flux.params(model), train_loader, opt)
 
  loss_train[epoch] = trainingloss()
  loss_valid[epoch] = validationloss()
  
  println("Training epoch No.",epoch,", loss_train = ", loss_train[epoch])
  println("Training epoch No.",epoch,", loss_valid = ", loss_valid[epoch])

end

# BSON.@save string("../Results/model_All_epoch",string(Nepochs),".bson") model


