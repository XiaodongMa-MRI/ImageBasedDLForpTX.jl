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

NVol = 36 # number of volumes
idx_valid = 2


# Unet define
channels_out=1

Nch_init = 16
kernelsize = (3, 3)


mf = matopen("../../Data/Data_processed_36Vol_NoCrop.mat")
Xtrain = read(mf, "Xtrain");
Ytrain = read(mf, "Ytrain");
close(mf)

for idx_subj = 1:size(Xtrain,5)
    Xtrain[:,:,:,:,idx_subj] = Xtrain[:,:,:,:,idx_subj]./maximum(Xtrain[:,:,:,:,idx_subj]);
    Ytrain[:,:,:,:,idx_subj] = Ytrain[:,:,:,:,idx_subj]./maximum(Ytrain[:,:,:,:,idx_subj]);
end

XValid = Xtrain[:,:,:,:,idx_valid];
YValid = Ytrain[:,:,:,:,idx_valid];

Xtrain = nothing;
Ytrain = nothing;

mf = matopen("../../Data/T1wBrain_processed_NoCrop.mat")
T1wBrain_resize_ds = read(mf, "T1wBrain_resize_ds");
close(mf)

for idx_subj = 1:size(T1wBrain_resize_ds,4)
    T1wBrain_resize_ds[:,:,:,idx_subj] = T1wBrain_resize_ds[:,:,:,idx_subj]./maximum(T1wBrain_resize_ds[:,:,:,idx_subj]);
end

Sall = size(XValid);
XValid = reshape(XValid,Sall[1],Sall[2],1,Sall[3]*Sall[4]);
YValid = reshape(YValid,Sall[1],Sall[2],1,Sall[3]*Sall[4]);

XValid_new = zeros(Float32,size(XValid,1),size(XValid,1),2,size(XValid,4));
XValid_new[:,:,1,:] = XValid;
XValid_new[:,:,2,:] = reshape(repeat(T1wBrain_resize_ds[:,:,:,idx_valid],1,1,NVol),size(XValid,1),size(XValid,1),1,size(XValid,4));

XValid = XValid_new;
XValid_new=nothing;


#XValid = JLD.load("Data/XYtrain_XYvalid_processed_36Vol_T1w.jld","XValid")
#YValid = JLD.load("Data/XYtrain_XYvalid_processed_36Vol_T1w.jld","YValid")

Nslc = 173
Nvol = 36

Nim = size(XValid,1);
#XValid = reshape(XValid,size(XValid,1),size(XValid,2),size(XValid,3),Nslc,Nvol)
XValid = reshape(XValid[:,:,:,:],size(XValid,1),size(XValid,2),2,Nslc,Nvol);
YValid = reshape(YValid,size(YValid,1),size(YValid,2),Nslc,Nvol);



#TestModelFile = "model_processed_36Vol_T1W_L2_SmallUNet30_k5_s14.bson"
#TestModelFile = "model_processed_36Vol_T1W_L2_SmallUNetGlobalSkip30_k5_s14.bson"
#TestModelFile = "model_EncoderDecoder_2345_1_noT1W.bson"
#TestModelFile = "model_EncoderDecoder_2345_1_T1W.bson"
#TestModelFile = "model_EncoderDecoderGlobalSkip_2345_1_T1W.bson"
#TestModelFile = "model_EncoderDecoderConvDownGlobalSkip_2345_1_T1WEpoch13.bson"
#TestModelFile = "model_EncoderDecoderConvDownReslearn_2345_1_T1WEpoch9.bson"
TestModelFile = "model_NCV_k4_epoch16.bson"
BSON.@load string("FromMSI/NestedCV/Results/",TestModelFile) model
channels_in=2

XtestOutput4write = zeros(size(YValid));

Xtest = zeros(Float32,size(XValid,1),size(XValid,2),channels_in,1);
#Ytest = zeros(Float32,size(XValid,1),size(XValid,2),1,1)
for idx_dir_test in 1:size(XValid,5)
  for idx_slc_test in 1:size(XValid,4)
    Xtest[:,:,:,1] = XValid[:,:,1:channels_in,idx_slc_test,idx_dir_test];
    #Ytest[:,:,1,1] = YValid[:,:,idx_slc_test,idx_dir_test]
#    XtestOutput4write[:,:,idx_slc_test,idx_dir_test] = model(Xtest);
    XtestOutput4write[:,:,idx_slc_test,idx_dir_test] = model(Xtest).+Xtest[:,:,1]
  end
end

file = matopen(string("FromMSI/Results/CV/",TestModelFile[1:end-5],".mat"), "w")
write(file, "XtestOutput4write", XtestOutput4write)
close(file)


