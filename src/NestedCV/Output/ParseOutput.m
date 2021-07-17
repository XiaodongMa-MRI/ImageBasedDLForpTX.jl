% parse the output
clear

%%
fall = dir(pwd);
Nsample = 50;
Nepoch = 30;
Validation_loss = zeros(Nepoch,Nsample,20);
for idx = 1:numel(fall)-2
    if contains(fall(idx+2).name,'.txt')

        fileID = fopen(fall(idx+2).name);
        ns = 1;
        while ns <= Nsample
            ne = 1;
            while ne <= Nepoch
                tline = fgetl(fileID);
                if contains(tline,'loss_valid = ')
                    Validation_loss(ne,ns,idx) = str2double(tline(35:end));
                    ne = ne+1;
                end
            end
            ns = ns+1;
        end
        fclose(fileID);
    end

end
%%
Validation_loss = reshape(Validation_loss,30,50,4,5);
Validation_loss_min = squeeze(min(Validation_loss(11:end,:,:,:),[],1));

mean_k1=mean(Validation_loss_min(:,:,1),2);
mean_k2=mean(Validation_loss_min(:,:,2),2);
mean_k3=mean(Validation_loss_min(:,:,3),2);
mean_k4=mean(Validation_loss_min(:,:,4),2);
mean_k5=mean(Validation_loss_min(:,:,1),2);

[loss_min_k1, idx_k1] = min(mean_k1) %13
[loss_min_k2, idx_k2] = min(mean_k2) %43
[loss_min_k3, idx_k3] = min(mean_k3) %5
[loss_min_k4, idx_k4] = min(mean_k4) %8
[loss_min_k5, idx_k5] = min(mean_k5) %13

%%
Train_all = cat(3,Training_loss_k1,Training_loss_k2,Training_loss_k3,Training_loss_k4,Training_loss_k5);
Train_all_mean=squeeze(mean(Train_all,2));
Valid_all = cat(3,Validation_loss_k1,Validation_loss_k2,Validation_loss_k3,Validation_loss_k4,Validation_loss_k5);
Valid_all_mean=squeeze(mean(Valid_all,2));

[~,IdxSort] = sort(Valid_all_mean(:,1));

figure;plot(Train_all_mean(IdxSort,:))
hold on
plot(Valid_all_mean(IdxSort,:),'--')

legend('Train k1','Train k2','Train k3','Train k4','Train k5','Valid k1','Valid k2','Valid k3','Valid k4','Valid k5')

%%
Loss_CV_valid_n8=[0.00011190007,8.8066954e-5,5.5016433e-5,8.1759885e-5,9.82311e-5];Loss_CV_train_n8=[4.3771928e-5,4.5263016e-5,4.8990754e-5,4.5356574e-5,5.0519884e-5];
bar([Loss_CV_train_n8;Loss_CV_valid_n8].')
% manually adjust the yaxis to 0 - 1.5e-4
legend('Training MSE loss','Testing MSE loss')
