# number of pooling levels
NpoolList = zeros(Int,50);
NchList = zeros(Int,50);
bs_list = zeros(Int,50);
li_list = zeros(Float32,50);
ldf_list = zeros(Float32,50);
lds_list = zeros(Float32,50);

for i in 1:50
    NpoolList[i] = rand((2,3,4))
    NchList[i] = rand((16,20,24,28,32))
    bs_list[i] = rand((16,20,24,28,32))
    li_list[i] = rand()*(0.01-0.0001)+0.0001
    lds_list[i] = rand((2,4,6,8))
    ldf_list[i] = rand()*(0.3-0.1)+0.1
end
JLD.save("HyperParsN50ForDeEncoder.jld","NpoolList",NpoolList,"NchList",NchList,"bs_list",bs_list,"li_list",li_list,"ldf_list",ldf_list,"lds_list",lds_list);


