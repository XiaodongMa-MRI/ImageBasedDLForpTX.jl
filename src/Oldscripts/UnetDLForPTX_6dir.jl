using UNet
using Flux
using Zygote
using MAT
#using Plots
using Statistics
using LinearAlgebra
using Random
using BSON
using JLD
using PyPlot


# load variables
mf = matopen("/home/naxos2-raid1/maxiao/Projects/pTXDL/JuliaPrograms/ImageBasedDLForpTX/test/Data_processed.mat")
Xtrain = read(mf, "Xtrain")
Ytrain = read(mf, "Ytrain")
close(mf)

#=
fig, ax = plt.subplots()
PyPlot.gray()
ax.imshow(abs.(Xtrain[:,:,50,1,1]))
#display(fig)
fig2, ax2 = plt.subplots()
PyPlot.gray()
ax2.imshow(abs.(Ytrain[:,:,50,1,1]))
#display(fig)
=#

Xtrain = Xtrain./maximum(maximum(maximum(Xtrain)))
Ytrain = Ytrain./maximum(maximum(maximum(Ytrain)))
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


JLD.save("XYtrain_XYvalid_processed.jld","Xtrain",Xtrain,"Ytrain",Ytrain,"XValid",XValid,"YValid",YValid)

#XValid = nothing;
#YValid = nothing;

#Xtrain = JLD.load("XYtrain_subj1_XYvalid_subj2_1shell_AP.jld","Xtrain")
#Ytrain = JLD.load("XYtrain_subj1_XYvalid_subj2_1shell_AP.jld","Ytrain")


idx_shuffle = shuffle(collect(1:size(Xtrain,4)))
Xtrain = Xtrain[:,:,:,idx_shuffle]
Ytrain = Ytrain[:,:,:,idx_shuffle]

Sall = size(XValid)
XValid = reshape(XValid,Sall[1],Sall[2],1,Sall[3]*Sall[4])
YValid = reshape(YValid,Sall[1],Sall[2],1,Sall[3]*Sall[4])


#=
Xtrain = Xtrain[:,:,1,1:round(Int,size(Xtrain,4)/2)]
Xtrain = reshape(Xtrain,size(Xtrain,1),size(Xtrain,2),1,size(Xtrain,3))
Ytrain = Ytrain[:,:,1,1:round(Int,size(Ytrain,4)/2)]
Ytrain = reshape(Ytrain,size(Ytrain,1),size(Ytrain,2),1,size(Ytrain,3))
=#

train_loader = Flux.Data.DataLoader(Xtrain,Ytrain, batchsize=10,shuffle=true)

u = Unet(1)

loss(x,y) = Flux.mse(u(x),y)
validationloss() = mean(Flux.mse(u(XValid[:,:,:,(idx-1)*10+1:idx*10]),YValid[:,:,:,(idx-1)*10+1:idx*10]) for idx=1:round(Int,size(XValid,4)/10))

cb = Flux.throttle(() -> @show(validationloss()), 10)
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
loss_valid[1] = validationloss()
for epoch in 2:10
  #@time Flux.train!(loss, Flux.params(u), train_loader, opt, cb = cb)
  @time my_custom_train!(loss, Flux.params(u), train_loader, opt)
  BSON.@save string("model_unet_ptx_processed_7dir_epoch",string(epoch),".bson") u
  loss_valid[epoch] = validationloss()
end



#test
XValid = JLD.load("XYtrain_XYvalid_processed.jld","XValid")
YValid = JLD.load("XYtrain_XYvalid_processed.jld","YValid")

Xtest = zeros(Float32,size(XValid,1),size(XValid,2),1,1)
Ytest = zeros(Float32,size(XValid,1),size(XValid,2),1,1)

idx_dir_test = 2
idx_slc_test = 15
Xtest[:,:,1,1] = XValid[:,:,idx_slc_test,idx_dir_test]
Ytest[:,:,1,1] = YValid[:,:,idx_slc_test,idx_dir_test]
Xtest_output = u(Xtest)

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




BSON.@load string("model_unet_ptx_processed_7dir_epoch",string(10),".bson") u
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



slc_list = [10 25]
dir_list = [1 2]
XtestOutput4write_slc10_25_dir1_2 = zeros(Float32,size(XValid,1),size(XValid,2),length(slc_list),length(dir_list),10)
for idx_epoch in 1:10
  BSON.@load string("model_unet_ptx_processed_7dir_epoch",string(idx_epoch),".bson") u
  Xtest = zeros(Float32,size(XValid,1),size(XValid,2),1,1)
  Ytest = zeros(Float32,size(XValid,1),size(XValid,2),1,1)
  for idx_dir_test in 1:length(dir_list)
    for idx_slc_test in 1:length(slc_list)
      Xtest[:,:,1,1] = XValid[:,:,slc_list[idx_slc_test],dir_list[idx_dir_test]]
      XtestOutput4write_slc10_25_dir1_2[:,:,idx_slc_test,idx_dir_test,idx_epoch] = u(Xtest)
    end
  end
end


file = matopen("XValidOutput_subj5_processed_7dir_epoch10.mat", "w")
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
