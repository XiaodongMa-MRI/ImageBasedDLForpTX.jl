using UNet
using Flux
using Zygote
using MAT
using Statistics
using LinearAlgebra
using Random
using BSON
using JLD
using PyPlot

#=
# load variables
mf = matopen("/home/naxos2-raid1/maxiao/Projects/pTXDL/Data_raw.mat")
Xtrain_AP_1 = read(mf, "Xtrain_AP_1")
Xtrain_AP_2 = read(mf, "Xtrain_AP_2")
Ytrain_AP_1 = read(mf, "Ytrain_AP_1_rot")
Ytrain_AP_2 = read(mf, "Ytrain_AP_2_rot")
close(mf)

#=
fig, ax = plt.subplots()
ax.imshow(abs.(Xtrain_AP_1[:,:,50,1,1]))
display(fig)
ax.imshow(abs.(Ytrain_AP_1[:,:,50,1,1]))
display(fig)
=#
Xtrain_AP_1_subj1 = Xtrain_AP_1[:,:,:,:,1]./maximum(maximum(maximum(Xtrain_AP_1[:,:,:,1,1])))
Xtrain_AP_1_subj2 = Xtrain_AP_1[:,:,:,:,2]./maximum(maximum(maximum(Xtrain_AP_1[:,:,:,1,2])))
Xtrain_AP_1_subj3 = Xtrain_AP_1[:,:,:,:,3]./maximum(maximum(maximum(Xtrain_AP_1[:,:,:,1,3])))
Xtrain_AP_2_subj1 = Xtrain_AP_2[:,:,:,:,1]./maximum(maximum(maximum(Xtrain_AP_1[:,:,:,1,1])))
Xtrain_AP_2_subj2 = Xtrain_AP_2[:,:,:,:,2]./maximum(maximum(maximum(Xtrain_AP_1[:,:,:,1,2])))
Xtrain_AP_2_subj3 = Xtrain_AP_2[:,:,:,:,3]./maximum(maximum(maximum(Xtrain_AP_1[:,:,:,1,3])))
Xtrain_AP_1 = nothing
Xtrain_AP_2 = nothing

Ytrain_AP_1_subj1 = Ytrain_AP_1[:,:,:,:,1]./maximum(maximum(maximum(Ytrain_AP_1[:,:,:,1,1])))
Ytrain_AP_1_subj2 = Ytrain_AP_1[:,:,:,:,2]./maximum(maximum(maximum(Ytrain_AP_1[:,:,:,1,2])))
Ytrain_AP_1_subj3 = Ytrain_AP_1[:,:,:,:,3]./maximum(maximum(maximum(Ytrain_AP_1[:,:,:,1,3])))
Ytrain_AP_2_subj1 = Ytrain_AP_2[:,:,:,:,1]./maximum(maximum(maximum(Ytrain_AP_1[:,:,:,1,1])))
Ytrain_AP_2_subj2 = Ytrain_AP_2[:,:,:,:,2]./maximum(maximum(maximum(Ytrain_AP_1[:,:,:,1,2])))
Ytrain_AP_2_subj3 = Ytrain_AP_2[:,:,:,:,3]./maximum(maximum(maximum(Ytrain_AP_1[:,:,:,1,3])))
Ytrain_AP_1 = nothing
Ytrain_AP_2 = nothing

#Xtrain = cat(Xtrain_AP_1_subj1,Xtrain_AP_2_subj1,Xtrain_AP_1_subj2,Xtrain_AP_2_subj2,dims=4)
Xtrain = cat(Xtrain_AP_1_subj1,Xtrain_AP_2_subj1,dims=4)
Sall = size(Xtrain)
Xtrain = reshape(Xtrain,Sall[1],Sall[2],1,Sall[3]*Sall[4])

#Ytrain = cat(Ytrain_AP_1_subj1,Ytrain_AP_2_subj1,Ytrain_AP_1_subj2,Ytrain_AP_2_subj2,dims=4)
Ytrain = cat(Ytrain_AP_1_subj1,Ytrain_AP_2_subj1,dims=4)
Sall = size(Ytrain)
Ytrain = reshape(Ytrain,Sall[1],Sall[2],1,Sall[3]*Sall[4])

XValid = cat(Xtrain_AP_1_subj3,Xtrain_AP_2_subj3,dims=4)
Sall = size(XValid)
XValid = reshape(XValid,Sall[1],Sall[2],1,Sall[3]*Sall[4])

YValid = cat(Ytrain_AP_1_subj3,Ytrain_AP_2_subj3,dims=4)
Sall = size(YValid)
YValid = reshape(YValid,Sall[1],Sall[2],1,Sall[3]*Sall[4])

Xtrain_AP_1_subj1 = nothing;
Xtrain_AP_1_subj2 = nothing;
Xtrain_AP_1_subj3 = nothing;
Xtrain_AP_2_subj1 = nothing;
Xtrain_AP_2_subj2 = nothing;
Xtrain_AP_2_subj3 = nothing;
Ytrain_AP_1_subj1 = nothing;
Ytrain_AP_1_subj2 = nothing;
Ytrain_AP_1_subj3 = nothing;
Ytrain_AP_2_subj1 = nothing;
Ytrain_AP_2_subj2 = nothing;
Ytrain_AP_2_subj3 = nothing;

idx_shuffle = shuffle(collect(1:size(Xtrain,4)))
Xtrain = Xtrain[:,:,:,idx_shuffle]
Ytrain = Ytrain[:,:,:,idx_shuffle]

JLD.save("XYtrain_subj1.jld","Xtrain",Xtrain,"Ytrain",Ytrain,"XValid",XValid,"YValid",YValid)
=#

Xtrain = JLD.load("XYtrain_subj1.jld","Xtrain")
Ytrain = JLD.load("XYtrain_subj1.jld","Ytrain")

train_loader = Flux.Data.DataLoader(Xtrain,Ytrain, batchsize=25,shuffle=true)

u = Unet(1)

loss(x,y) = Flux.mse(u(x),y)
validationloss() = Flux.mse(u(XValid),YValid)

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
  
  

for epoch in 1:1
#@time Flux.train!(loss, Flux.params(u), train_loader, opt, cb = cb)
@time my_custom_train!(loss, Flux.params(u), train_loader, opt)
end
