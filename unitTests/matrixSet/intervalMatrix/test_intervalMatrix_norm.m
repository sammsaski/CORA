function res = test_intervalMatrix_norm
% test_intervalMatrix_norm - unit test function of norm; the result is
%    compared to the norm of some vertices
% 
% Syntax:  
%    res = test_matZonotope_norm
%
% Inputs:
%    -
%
% Outputs:
%    res - boolean 
%
% Example: 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% Author:       Matthias Althoff
% Written:      02-November-2017
% Last update:  ---
% Last revision:---

%------------- BEGIN CODE --------------

% create interval matrix
% center
matrixCenter = [ ...
0.1030783856804509, -0.9101278715321481, 0.6053092778292544, -0.7608595779453735, -0.2653784107456929, 0.8833290280791213, -0.5867093811050932, -0.3920964615009785, -0.9330031960700655, 0.2224896347355838; ...
-0.3358623384378412, -0.6606199258835557, 0.3453023581373951, 0.3373340161562055, 0.1728220584061801, 0.0571996064779656, -0.3622320218412414, -0.6610674364129314, 0.8292507232278290, -0.6361200702133358; ...
0.7258992470204615, 0.2226582101541923, 0.4569764282907689, -0.8375957540168222, 0.9482367162906631, 0.5867543962581900, 0.8899927549727960, -0.0438826364149949, -0.0382136940774347, 0.4472851749922520; ...
-0.3046165455476422, 0.9005982283328562, -0.6360513364531990, 0.4124682114261358, -0.8080476776070364, -0.5728913883303508, -0.5916024282199834, -0.5937651381699391, 0.6597700848648822, -0.0758241326363369; ...
0.7096807070659510, 0.4998074647484749, 0.9713758502353100, 0.9376564546258963, 0.2457649012001917, 0.8577861212555120, 0.5407432738095963, 0.4696139508867379, 0.2178340806678498, 0.6784418775414429; ...
-0.7440856538867173, -0.5786954854131867, -0.7372557698893745, 0.6013975816933068, -0.7329906793875653, -0.6534173712962281, 0.8794343222311480, -0.8762545300911007, -0.9999624970157370, -0.8504824917103764; ...
0.9084850705727758, -0.8989100454650638, 0.9920886519204546, 0.9198930027855210, 0.1288885207530313, 0.3430266672898845, 0.0650526293251279, 0.9152025750401234, -0.2713856840944358, -0.2328442840941238; ...
-0.5748584958454852, 0.7413066606288192, -0.3728964105823804, -0.4680697157913984, 0.1567639105156624, 0.0153859735195778, 0.3410331006983629, 0.1066855250868497, 0.3867543050330005, 0.1620966903388332; ...
0.8838491015893175, 0.9233098306708436, -0.9715289785483805, -0.6531267294643079, 0.5304519869096429, -0.5966071048661272, 0.8169855310441498, -0.5101349852246480, 0.5764394251020426, 0.5059990141202664; ...
-0.2456725263694224, -0.9851975999597997, -0.1965669055496231, 0.4875990931423375, 0.7099594112562833, -0.2178327228088248, -0.2044138432022624, 0.0436619915851515, -0.5099717870717175, 0.4996095777778413];

% delta
matrixDelta = [ ...
0.5578402240821573, 0.4480594631659266, 0.4074035017560966, 0.6771853549136722, 0.3543348838584117, 0.6441100952452690, 0.8495355582288353, 0.6278255744150814, 0.1436933992616499, 0.3351861257547281; ...
0.6896205123608918, 0.6221694039460455, 0.2048463746074216, 0.9862023306825124, 0.5799116825371671, 0.2440423838466353, 0.4726200288848037, 0.4232749194458143, 0.6006456898620373, 0.1138120903389709; ...
0.3199877866231424, 0.4216791412865608, 0.7827239258709241, 0.7270249408196269, 0.4073204743518910, 0.1826154452706400, 0.3491414686876024, 0.4444706725212727, 0.2858064385778392, 0.4527258512300267; ...
0.1975405021138973, 0.3669594034158065, 0.4586363546463044, 0.3111162959953676, 0.2679696978160009, 0.0291491798456658, 0.6200252911535523, 0.0028626563357185, 0.3435967110843504, 0.5710127416120387; ...
0.9793349810725723, 0.0490281291269694, 0.9324143437054532, 0.6517805698384385, 0.1078504805890825, 0.8670477241769642, 0.3850762552058492, 0.1069660193089993, 0.7927806156356799, 0.0439665433745674; ...
0.3629887146602292, 0.5749778125942429, 0.7414479136692824, 0.4566946301087875, 0.0382727176435640, 0.0890413584653672, 0.1217506599194927, 0.3617780179631745, 0.2507133266029168, 0.4765692269314052; ...
0.5816047528676512, 0.1491810230716484, 0.9241603872800250, 0.6504155066127847, 0.8648710999231712, 0.3441508894129642, 0.3908836656561908, 0.0782180721778618, 0.1386923529769383, 0.9975646502380379; ...
0.9165581749947073, 0.1223512040570282, 0.7902486311751354, 0.8614372801117565, 0.6490311489425237, 0.3718782459650920, 0.2149805751386201, 0.8430152525821835, 0.9114590240376715, 0.5310897471367274; ...
0.9659947386083864, 0.3649494588467126, 0.3502137362246930, 0.7050593708594525, 0.0182220543679318, 0.0176254123072940, 0.7443216159147799, 0.3702398673139704, 0.0368920866588464, 0.8946484402551380; ...
0.3540536342295094, 0.8794902052149960, 0.2578886467521072, 0.4026839622341357, 0.3425760804223682, 0.4395643811638824, 0.1385713550603721, 0.2429307149261678, 0.3918102674659337, 0.1527339386045851];

% instantiate interval matrix
M_int = intervalMatrix(matrixCenter, matrixDelta);

% obtain result of some vertices-----------------------
V = randomSampling(M_int,10);

%loop through vertices
n_1_sample = zeros(length(V),1);
n_2_sample = n_1_sample;
n_inf_sample = n_1_sample;
for i=1:length(V)
    n_1_sample(i) = norm(V{i}, 1);
    n_2_sample(i) = norm(V{i}, 2);
    n_inf_sample(i) = norm(V{i}, inf);
end
%-------------------------------------------------------

% obtain results
n_1 = norm(M_int, 1);
n_inf = norm(M_int, inf);

%check if slightly bloated results enclose others
res_1 = all(n_1_sample <= n_1*1+1e-8);
res_2 = all(n_inf_sample <= n_inf*1+1e-8);

%result of different computation techniques
res = res_1 && res_2;

% save result if random test failed
if ~res
     file_name = strcat('test_intervalMatrix_norm_', ...
                             datestr(now,'mm-dd-yyyy_HH-MM'));
                  
     file_path = fullfile(CORAROOT, 'unitTests', 'failedTests', file_name);
     save(file_path, 'V');
end

%------------- END OF CODE --------------