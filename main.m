%%%%%%%%%%%%%%%%%%%%%% READ from .mat files and perform alignment if necesary %%%%%%%%%%%%
%% Routine #1 Load files
% Set the file index to be read; file index is appeneded to the file name
clear all;
close all;
sample_rate = 4800;
live_durationn = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_counter = 1;
participant_name = 'Sunaal';
curr_angle = 0;
routine = 'R1_MVC';
file_name = 'maximum_contraction_';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % file names for torque
file_name_read = ['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/' file_name num2str(file_counter) '.mat'];
load(file_name_read);

gemg_scale = 1000; % Output units per 1mV
vmo_emg_raw = vmo_emg_raw ./gemg_scale;
vmo_emg_data = vmo_emg_data ./gemg_scale;
qf1_emg_raw = qf1_emg_raw ./gemg_scale;
qf1_emg_data = qf1_emg_data ./gemg_scale;
qf2_emg_raw = qf2_emg_raw ./gemg_scale;
qf2_emg_data = qf2_emg_data ./gemg_scale;
vl_emg_raw = vl_emg_raw ./gemg_scale;
vl_emg_data = vl_emg_data ./gemg_scale;
%%
%%%%%%%%%%%%%%%% ALIGNMENT %%%%%%%%%%%%%%%%%%%
willBeAligned = 0;

if willBeAligned
    delta_t = ginput;
    delay = delta_t(2,1) - delta_t(1,1);

    % Emg lagging = -1; Emg leading = 1
    lag_factor = -1;

    start_time_emg = 40;
    stop_time_emg = 120;

    start_time_torque = delay + lag_factor*start_time_emg ; 
    stop_time_torque = delay + llag_factor*stop_time_emg;
    % Find corresponding indeces for start and stop times for EMG and TOrque
    start_index_torque = 1;
    stop_index_torque = length(torque_data);
    start_index_emg = 1;
    stop_index_emg = length(vmo_emg_data);
    for i = 1:length(torque_data)
        if(torque_data(i,1) >= start_time_torque )
            start_index_torque = i;
            break;
        end
    end
    for i = start_index_torque:length(torque_data)
        if(torque_data(i,1) >= stop_time_torque )
            stop_index_torque = i;
            break;
        end
    end
    for i = 1:length(vmo_emg_data)
        if(vmo_emg_data(i,1) >= start_time_emg )
            start_index_emg = i;
            break;
        end
    end
    for i = start_index_emg:length(vmo_emg_data)
        if(vmo_emg_data(i,1) >= stop_time_emg )
            stop_index_emg = i;
            break;
        end
    end

    % create time series for emg and torque - we will consider emg time samples
    % as the standard time sampling for both emg and torque
    % This is a requirement only for gAmp, due to asynchronous sampling between
    % the DAQ and gAmp
    time_samples = vmo_emg_data(start_index_emg:stop_index_emg,1);
    % Trim data based on start and stop times
    vmo_emg_raw = vmo_raw_gemg((start_index_emg:stop_index_emg),2);
    vmo_emg_data = vmo_hp_gemg((start_index_emg:stop_index_emg),2);
    qf1_emg_raw = qf1_raw_gemg((start_index_emg:stop_index_emg),2);
    qf1_emg_data = qf1_hp_gemg((start_index_emg:stop_index_emg),2);
    qf2_emg_raw = qf2_raw_gemg((start_index_emg:stop_index_emg),2);
    qf2_emg_data = qf2_hp_gemg((start_index_emg:stop_index_emg),2);
    vl_emg_raw = vl_raw_gemg((start_index_emg:stop_index_emg),2);
    vl_emg_data = vl_hp_gemg((start_index_emg:stop_index_emg),2);
    torque_data = torque_raw((start_index_torque:stop_index_torque),2);
    torque_data = torque_data(1:length(vmo_emg_data));
end

%% PLOT DATA TO VERIFY READ
close all;

% VMO
figure();
subplot(3,1,1);
grid on;
plot(time_samples,vmo_emg_raw);
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'));
title('VMO: Raw EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,2);
plot(time_samples,vmo_emg_data);
ylim([-5 5]);
title('VMO: High Passed (18Hz) EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,3);
plot(time_samples,torque_data);
ylim([-500 500]);
title('Raw Force of desired time period');
xlabel('Time[s]');
ylabel('Force[N]');

% QF1
figure();
subplot(3,1,1);
grid on;
plot(time_samples,qf1_emg_raw);
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'));
title('QF1: Raw EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,2);
plot(time_samples,qf1_emg_data);
ylim([-5 5]);
title('QF1: High Passed (18Hz) EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,3);
plot(time_samples,torque_data);
ylim([-500 500]);
title('Raw Force of desired time period');
xlabel('Time[s]');
ylabel('Force[N]');

% QF2
figure();
subplot(3,1,1);
grid on;
plot(time_samples,qf2_emg_raw);
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'));
title('QF2: Raw EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,2);
plot(time_samples,qf2_emg_data);
ylim([-5 5]);
title('QF2: High Passed (18Hz) EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,3);
plot(time_samples,torque_data);
ylim([-500 500]);
title('Raw Force of desired time period');
xlabel('Time[s]');
ylabel('Force[N]');

% VL
figure();
grid on;
subplot(3,1,1);
plot(time_samples,vl_emg_raw);
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'))
title('VL: Raw EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,2);
plot(time_samples,vl_emg_data);
ylim([-5 5]);
title('VL: High Passed (18Hz) EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,3);
plot(time_samples,torque_data);
ylim([-500 500]);
title('Raw Force of desired time period');
xlabel('Time[s]');
ylabel('Force[N]');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MUSCLE TYPE %%%%%%%%%%%%%%%%%%%%%%%%%%%%
muscle = 'VL';
emg_data = vl_emg_data; %% high pass filtered
emg_raw = vl_emg_raw; %% raw data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   DEAVERAGING EMG DATA

% de-averaging for frequency analysis; remove offset to eliminate power at
% low frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%% RAW EMG 
voltage_zero_contraction = mean(emg_raw);
emg_deavg_raw = zeros(length(emg_raw),1);
for i=1:length(emg_raw)
    emg_deavg_raw(i) = emg_raw(i) - voltage_zero_contraction;
end
figure();
plot(time_samples,emg_raw);
hold on;
plot(time_samples,emg_deavg_raw);
hold off;
title([muscle ': Deaveraged on EMG RAW data']);
xlabel('Time[s]');
ylabel('Amplitude[mV]');
legend('EMG Original','EMG Deaveraged');
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_RAW_deavg_emg_' num2str(file_counter)]);
%%%%%%%%%%%%%%%%%%%%%%%%%%% HP EMG
voltage_zero_contraction =  mean(emg_data);
emg_deavg_hp = zeros(length(emg_data),1);
for i=1:length(emg_data)
    emg_deavg_hp(i) = emg_data(i) - voltage_zero_contraction;
end
figure();
plot(time_samples,emg_data);
hold on;
plot(time_samples,emg_deavg_hp);
hold off;
title([muscle ': Deaveraged on EMG HP data']);
xlabel('Time[s]');
ylabel('Amplitude[mV]');
legend('EMG Original','EMG Deaveraged');
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_HP_deavg_emg_' num2str(file_counter)]);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  BANDPASSING DEAVERAGED EMG DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
% Bandpass Butterworth dual - nth order filter
fc_lower = 15; % lower cut-off freq
fc_uper = 300; % upper cut-off freq
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bandpass_order = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[A,B,C,D] = butter(bandpass_order/2,[fc_lower fc_uper]/(sample_rate/2));
% Design the bandpass filter
d = designfilt('bandpassiir','FilterOrder',bandpass_order, ...
    'HalfPowerFrequency1',fc_lower,'HalfPowerFrequency2',fc_uper, ...
    'SampleRate',sample_rate);
sos = ss2sos(A,B,C,D);
fvt = fvtool(sos,d,'Fs',sample_rate);
legend(fvt,'butter','designfilt')

%%%%%%%%%%%%%%%%%%%%%%%%%%% RAW EMG
emg_filt_freq_spectrum_raw = filtfilt(d,emg_deavg_raw);
figure();
plot(time_samples,emg_deavg_raw);
hold on;
plot(time_samples,emg_filt_freq_spectrum_raw);
hold off;
title([muscle ': Bandpassed on EMG RAW data']);
xlabel('Time[s]');
ylabel('Amplitude[mV]');
legend('EMG Deaveraged','EMG Bandpassed');
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_RAW_bandpassed_emg_' num2str(file_counter)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%% HP EMG
emg_filt_freq_spectrum_hp = filtfilt(d,emg_deavg_hp);
figure();
plot(time_samples,emg_deavg_hp);
hold on;
plot(time_samples,emg_filt_freq_spectrum_hp);
hold off;
title([muscle ': Bandpassed on EMG HP data']);
xlabel('Time[s]');
ylabel('Amplitude[mV]');
legend('EMG Deaveraged','EMG Bandpassed');
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_RAW_bandpassed_emg_' num2str(file_counter)]);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  RECTIFYING FILTERED EMG DATA

% Full wave rectification for amplitude modulation for low pass butter
% filter with zero lag
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filter parameters for desired envelope dectection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
cuttoff_freq = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Butterworth filter LP for envelop dectection
wn = sample_rate/2; % nyquist frequency for digital data - max possible freq of data
norm_freq = cuttoff_freq/wn; % [rads/sample]
filter_order = 2; 
[a b]=butter(filter_order/2,norm_freq); %create butterworth coefficients 

% moving average filter for torque
windowSize = 10;
a2 = 1;
b2 = (1/windowSize)*ones(1,windowSize);
torque_filt_data = filter(b2,a2,torque_data);

%%%%%%%%%%%%%%%%%%%%%%%%%%% RAW EMG
voltage_zero_contraction = mean(emg_filt_freq_spectrum_raw);
emg_rect_raw = zeros(length(emg_filt_freq_spectrum_raw),1);
for i=1:length(emg_filt_freq_spectrum_raw)
    emg_rect_raw(i) = abs(emg_filt_freq_spectrum_raw(i) - voltage_zero_contraction);
end
figure();
plot(time_samples,emg_filt_freq_spectrum_raw);
hold on;
plot(time_samples,emg_rect_raw);
hold off;
title([muscle ': Full-wave rectification on EMG RAW data']);
xlabel('Time[s]');
ylabel('Amplitude[mV]');
legend('EMG Bandpassed','EMG Rectified');
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_RAW_rectified_emg_' num2str(file_counter)]);
% generating emg envelope via butterworth filtering
emg_filt_envelope_raw = filtfilt(a,b,emg_rect_raw); % filter the data with zero lag
figure();
plot(time_samples,emg_rect_raw);
hold on;
plot(time_samples,emg_filt_envelope_raw);
hold off;
title([muscle ': Envelope on EMG RAW data']);
xlabel('Time[s]');
ylabel('Amplitude[mV]');
legend('EMG Rectified','EMG Enveloped');
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_RAW_enveloped_emg_' num2str(file_counter)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%% HP EMG
voltage_zero_contraction = mean(emg_filt_freq_spectrum_hp);
emg_rect_hp = zeros(length(emg_filt_freq_spectrum_hp),1);
for i=1:length(emg_filt_freq_spectrum_hp)
    emg_rect_hp(i) = abs(emg_filt_freq_spectrum_hp(i) - voltage_zero_contraction);
end
figure();
plot(time_samples,emg_filt_freq_spectrum_hp);
hold on;
plot(time_samples,emg_rect_hp);
hold off;
title([muscle ':  Full-wave rectification on EMG HP data']);
xlabel('Time[s]');
ylabel('Amplitude[mV]');
legend('EMG Bandpassed','EMG Rectified');
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_RAW_rectified_emg_' num2str(file_counter)]);
% generating emg envelope via butterworth filtering
emg_filt_envelope_hp = filtfilt(a,b,emg_rect_hp); % filter the data with zero lag
figure();
plot(time_samples,emg_rect_hp);
hold on;
plot(time_samples,emg_filt_envelope_hp);
hold off;
title([muscle ': Envelope on EMG RAW data']);
xlabel('Time[s]');
ylabel('Amplitude[mV]');
legend('EMG Rectified','EMG Enveloped');
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_RAW_enveloped_emg_' num2str(file_counter)]);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  ANALYSIS OF RELATIONSHIP BETWEEN FORCE AND EMG
%                     
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%% RAW EMG
% plot emg against torque - both filtered
figure();
subplot(2,1,1);
[ax,h1,h2] = plotyy(time_samples,emg_filt_envelope_raw, time_samples, torque_filt_data);
title([muscle ': EMG RAW: Filtered EMG and Force superimposed with respective y-scales']);
legend('EMG','Force');
subplot(2,1,2);
plot(emg_filt_envelope_raw,torque_data);
title([muscle ': Relationship between EMG and Force']);
xlabel('EMG [mV]');
ylabel('Force[N]');
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_torque_emg_' num2str(file_counter)]);

% Analysis of trends between emg and torque
mdl = fitlm(emg_filt_envelope_raw,torque_filt_data)

% Polynomial Fit - 1st order
[p,S] = polyfit(emg_filt_envelope_raw,torque_data,1);%compute a linear regression that predicts TORQUE from EMG
torque_curve_fit = polyval(p,emg_filt_envelope_raw); % evaluate the polynomial fit values for TORQUE with respect to EMG
yresid = torque_data - torque_curve_fit; % compute theresidual values as a vector of signed numbers
SSresid = sum(yresid.^2); % Square the residuals and total them to obtain the residual sum of squares
SStotal = (length(torque_data)-1) * var(torque_data); % Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1
Rsquared_1 = 1 - SSresid/SStotal % Rsquared value
figure();
plot(emg_filt_envelope_raw,torque_data,'x','MarkerSize',0.01);
hold on;
plot(emg_filt_envelope_raw,torque_curve_fit,'LineWidth',5);
ylim([0 500]);
title([ muscle ' - Polynomial fit 1st order: R\^2 = ' num2str(Rsquared_1)]);
eqn = [num2str(p(1)) 'x + ' num2str(p(2))];
text(10,300,eqn);
xlabel('EMG [mV]');
ylabel('Force[N]');
%%%%%%%%%%%%%%%%%% APPEND POLYNOMIAL FOR FINAL TORQUE-EMG ANALYSIS %%%%%%%%
dlmwrite(['data/' participant_name '/' muscle '_emg_raw_polynomial_1st_order.csv'], ...
         p, 'delimiter',',','-append');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_polyfit_1st_order_' num2str(file_counter)]);

% Polynomial Fit - 2nd order
[p,S] = polyfit(emg_filt_envelope_raw,torque_data,2); %compute a linear regression that predicts TORQUE from EMG
torque_curve_fit = polyval(p,emg_filt_envelope_raw); % evaluate the polynomial fit values for TORQUE with respect to EMG
yresid = torque_data - torque_curve_fit; % compute theresidual values as a vector of signed numbers
SSresid = sum(yresid.^2); % Square the residuals and total them to obtain the residual sum of squares
SStotal = (length(torque_data)-1) * var(torque_data); % Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1
Rsquared_2 = 1 - SSresid/SStotal % Rsquared value
figure();
plot(emg_filt_envelope_raw,torque_data,'x','MarkerSize',0.01);
hold on;
plot(emg_filt_envelope_raw,torque_curve_fit,'LineWidth',5);
ylim([0 500]);
title([muscle ' - Polynomial fit 2nd order: R\^2 = ' num2str(Rsquared_2)]);
eqn = [num2str(p(1)) 'x^2 + ' num2str(p(2)) 'x + ' num2str(p(3))];
text(10,300,eqn);
xlabel('EMG [mV]');
ylabel('Force[N]');
%%%%%%%%%%%%%%%%%% APPEND POLYNOMIAL FOR FINAL TORQUE-EMG ANALYSIS %%%%%%%%
dlmwrite(['data/' participant_name '/' muscle '_emg_raw_polynomial_2nd_order.csv'], ...
         p, 'delimiter',',','-append');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_polyfit_2nd_order_' num2str(file_counter)]);

% Polynomial Fit - 3rd order
[p,S] = polyfit(emg_filt_envelope_raw,torque_data,3); %compute a linear regression that predicts TORQUE from EMG
torque_curve_fit = polyval(p,emg_filt_envelope_raw); % evaluate the polynomial fit values for TORQUE with respect to EMG
yresid = torque_data - torque_curve_fit; % compute theresidual values as a vector of signed numbers
SSresid = sum(yresid.^2); % Square the residuals and total them to obtain the residual sum of squares
SStotal = (length(torque_data)-1) * var(torque_data); % Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1
Rsquared_3 = 1 - SSresid/SStotal % Rsquared value
figure();
plot(emg_filt_envelope_raw,torque_data,'x','MarkerSize',0.01);
hold on;
plot(emg_filt_envelope_raw,torque_curve_fit,'LineWidth',5);
ylim([0 500]);
title([muscle ' - Polynomial fit 3rd order: R\^2 = ' num2str(Rsquared_3)]);
eqn = [num2str(p(1)) 'x^3 + ' num2str(p(2)) 'x^2 + ' num2str(p(3)) 'x + ' num2str(p(4))];
text(10,300,eqn);
xlabel('EMG [mV]');
ylabel('Force[N]');
%%%%%%%%%%%%%%%%%% APPEND POLYNOMIAL FOR FINAL TORQUE-EMG ANALYSIS %%%%%%%%
dlmwrite(['data/' participant_name '/' muscle '_emg_raw_polynomial_3rd_order.csv'], ...
         p, 'delimiter',',','-append');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_polyfit_3rd_order_' num2str(file_counter)]);

% Polynomial Fit - 4th order
[p,S] = polyfit(emg_filt_envelope_raw,torque_data,4); %compute a linear regression that predicts TORQUE from EMG
torque_curve_fit = polyval(p,emg_filt_envelope_raw); % evaluate the polynomial fit values for TORQUE with respect to EMG
yresid = torque_data - torque_curve_fit; % compute theresidual values as a vector of signed numbers
SSresid = sum(yresid.^2); % Square the residuals and total them to obtain the residual sum of squares
SStotal = (length(torque_data)-1) * var(torque_data); % Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1
Rsquared_4 = 1 - SSresid/SStotal % Rsquared value
figure();
plot(emg_filt_envelope_raw,torque_data,'x','MarkerSize',0.01);
hold on;
plot(emg_filt_envelope_raw,torque_curve_fit,'LineWidth',5);
ylim([0 500]);
title([muscle ' - Polynomial fit 4th order: R\^2 = ' num2str(Rsquared_4)]);
eqn = [num2str(p(1)) 'x^4 + ' num2str(p(2)) 'x^3 + ' num2str(p(3)) 'x^2 + ' num2str(p(4)) 'x + ' num2str(p(5))];
text(10,300,eqn);
xlabel('EMG [mV]');
ylabel('Force[N]');
%%%%%%%%%%%%%%%%%% APPEND POLYNOMIAL FOR FINAL TORQUE-EMG ANALYSIS %%%%%%%%
dlmwrite(['data/' participant_name '/' muscle '_emg_raw_polynomial_4th_order.csv'], ...
         p, 'delimiter',',','-append');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_polyfit_4th_order_' num2str(file_counter)]);


%%%%%%%%%%%%%%%%%%%%%%%%%%% HP EMG
% plot emg against torque - both filtered
figure();
subplot(2,1,1);
[ax,h1,h2] = plotyy(time_samples,emg_filt_envelope_hp, time_samples, torque_filt_data);
title([ muscle ' : EMG HP: Filtered EMG and Force superimposed with respective y-scales']);
legend('EMG','Force');
subplot(2,1,2);
plot(emg_filt_envelope_hp,torque_data);
title([ muscle ': Relationship between EMG and Force']);
xlabel('EMG [mV]');
ylabel('Force[N]');
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_torque_emg_' num2str(file_counter)]);

% Analysis of trends between emg and torque
mdl = fitlm(emg_filt_envelope_hp,torque_filt_data)

% Polynomial Fit - 1st order
[p,S] = polyfit(emg_filt_envelope_hp,torque_data,1);%compute a linear regression that predicts TORQUE from EMG
torque_curve_fit = polyval(p,emg_filt_envelope_hp); % evaluate the polynomial fit values for TORQUE with respect to EMG
yresid = torque_data - torque_curve_fit; % compute theresidual values as a vector of signed numbers
SSresid = sum(yresid.^2); % Square the residuals and total them to obtain the residual sum of squares
SStotal = (length(torque_data)-1) * var(torque_data); % Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1
Rsquared_1 = 1 - SSresid/SStotal % Rsquared value
figure();
plot(emg_filt_envelope_hp,torque_data,'x','MarkerSize',0.01);
hold on;
plot(emg_filt_envelope_hp,torque_curve_fit,'LineWidth',5);
ylim([0 500]);
title([muscle ' - Polynomial fit 1st order: R\^2 = ' num2str(Rsquared_1)]);
eqn = [num2str(p(1)) 'x + ' num2str(p(2))];
text(10,300,eqn);
xlabel('EMG [mV]');
ylabel('Force[N]');
%%%%%%%%%%%%%%%%%% APPEND POLYNOMIAL FOR FINAL TORQUE-EMG ANALYSIS %%%%%%%%
dlmwrite(['data/' participant_name '/' muscle '_emg_hp_polynomial_1st_order.csv'], ...
         p, 'delimiter',',','-append');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_polyfit_1st_order_' num2str(file_counter)]);

% Polynomial Fit - 2nd order
[p,S] = polyfit(emg_filt_envelope_hp,torque_data,2); %compute a linear regression that predicts TORQUE from EMG
torque_curve_fit = polyval(p,emg_filt_envelope_hp); % evaluate the polynomial fit values for TORQUE with respect to EMG
yresid = torque_data - torque_curve_fit; % compute theresidual values as a vector of signed numbers
SSresid = sum(yresid.^2); % Square the residuals and total them to obtain the residual sum of squares
SStotal = (length(torque_data)-1) * var(torque_data); % Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1
Rsquared_2 = 1 - SSresid/SStotal % Rsquared value
figure();
plot(emg_filt_envelope_hp,torque_data,'x','MarkerSize',0.01);
hold on;
plot(emg_filt_envelope_hp,torque_curve_fit,'LineWidth',5);
ylim([0 500]);
title([muscle ' - Polynomial fit 2nd order: R\^2 = ' num2str(Rsquared_2)]);
eqn = [num2str(p(1)) 'x^2 + ' num2str(p(2)) 'x + ' num2str(p(3))];
text(10,300,eqn);
xlabel('EMG [mV]');
ylabel('Force[N]');
%%%%%%%%%%%%%%%%%% APPEND POLYNOMIAL FOR FINAL TORQUE-EMG ANALYSIS %%%%%%%%
dlmwrite(['data/' participant_name '/' muscle '_emg_hp_polynomial_2nd_order.csv'], ...
         p, 'delimiter',',','-append');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_polyfit_2nd_order_' num2str(file_counter)]);

% Polynomial Fit - 3rd order
[p,S] = polyfit(emg_filt_envelope_hp,torque_data,3); %compute a linear regression that predicts TORQUE from EMG
torque_curve_fit = polyval(p,emg_filt_envelope_hp); % evaluate the polynomial fit values for TORQUE with respect to EMG
yresid = torque_data - torque_curve_fit; % compute theresidual values as a vector of signed numbers
SSresid = sum(yresid.^2); % Square the residuals and total them to obtain the residual sum of squares
SStotal = (length(torque_data)-1) * var(torque_data); % Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1
Rsquared_3 = 1 - SSresid/SStotal % Rsquared value
figure();
plot(emg_filt_envelope_hp,torque_data,'x','MarkerSize',0.01);
hold on;
plot(emg_filt_envelope_hp,torque_curve_fit,'LineWidth',5);
ylim([0 500]);
title([muscle ' - Polynomial fit 3rd order: R\^2 = ' num2str(Rsquared_3)]);
eqn = [num2str(p(1)) 'x^3 + ' num2str(p(2)) 'x^2 + ' num2str(p(3)) 'x + ' num2str(p(4))];
text(10,300,eqn);
xlabel('EMG [mV]');
ylabel('Force[N]');
%%%%%%%%%%%%%%%%%% APPEND POLYNOMIAL FOR FINAL TORQUE-EMG ANALYSIS %%%%%%%%
dlmwrite(['data/' participant_name '/' muscle '_emg_hp_polynomial_3rd_order.csv'], ...
         p, 'delimiter',',','-append');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_polyfit_3rd_order_' num2str(file_counter)]);

% Polynomial Fit - 4th order
[p,S] = polyfit(emg_filt_envelope_hp,torque_data,4); %compute a linear regression that predicts TORQUE from EMG
torque_curve_fit = polyval(p,emg_filt_envelope_hp); % evaluate the polynomial fit values for TORQUE with respect to EMG
yresid = torque_data - torque_curve_fit; % compute theresidual values as a vector of signed numbers
SSresid = sum(yresid.^2); % Square the residuals and total them to obtain the residual sum of squares
SStotal = (length(torque_data)-1) * var(torque_data); % Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1
Rsquared_4 = 1 - SSresid/SStotal % Rsquared value
figure();
plot(emg_filt_envelope_hp,torque_data,'x','MarkerSize',0.01);
hold on;
plot(emg_filt_envelope_hp,torque_curve_fit,'LineWidth',5);
ylim([0 500]);
title([muscle ' - Polynomial fit 4th order: R\^2 = ' num2str(Rsquared_4)]);
eqn = [num2str(p(1)) 'x^4 + ' num2str(p(2)) 'x^3 + ' num2str(p(3)) 'x^2 + ' num2str(p(4)) 'x + ' num2str(p(5))];
text(10,300,eqn);
xlabel('EMG [mV]');
ylabel('Force[N]');
%%%%%%%%%%%%%%%%%% APPEND POLYNOMIAL FOR FINAL TORQUE-EMG ANALYSIS %%%%%%%%
dlmwrite(['data/' participant_name '/' muscle '_emg_hp_polynomial_4th_order.csv'], ...
         p, 'delimiter',',','-append');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_polyfit_4th_order_' num2str(file_counter)]);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  99% of occupied Bandwidth: EMG
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%% RAQ EMG
x = emg_deavg_raw;
Fs = sample_rate;
figure;
obw(x,Fs);
[bw,flo,fhi,powr] = obw(x,Fs);
legend([ muscle ': Emg (RAW) Deaveraged']);
pcent = powr/bandpower(x)*100;
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_bw99_emg_deavg_' num2str(file_counter)]);
x = emg_filt_freq_spectrum_raw;
Fs = sample_rate;
figure;
obw(x,Fs);
[bw,flo,fhi,powr] = obw(x,Fs);
legend([ muscle ':Emg (RAW) Bandpassed']);
pcent = powr/bandpower(x)*100;
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_bw99_emg_filt_' num2str(file_counter)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%% HP EMG
x = emg_deavg_hp;
Fs = sample_rate;
figure;
obw(x,Fs);
[bw,flo,fhi,powr] = obw(x,Fs);
legend([ muscle ':Emg (HP) Deaveraged']);
pcent = powr/bandpower(x)*100;
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_bw99_emg_deavg_' num2str(file_counter)]);
x = emg_filt_freq_spectrum_raw;
Fs = sample_rate;
figure;
obw(x,Fs);
[bw,flo,fhi,powr] = obw(x,Fs);
legend([ muscle ':Emg (HP) Bandpassed']);
pcent = powr/bandpower(x)*100;
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_bw99_emg_filt_' num2str(file_counter)]);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Frequency analysis of EMG: Bode plots (phase and magnitude)
%             
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
Fs = sample_rate;

%%%%%%%%%%%%%%%%%%%%%%%%%%% RAW EMG
y = emg_deavg_raw;
NFFT = length(y);
Y = fft(y,NFFT);
F = ((0:1/NFFT:1-1/NFFT)*Fs).';
magnitudeY = abs(Y);        % Magnitude of the FFT
phaseY = unwrap(angle(Y));  % Phase of the FFT
helperFrequencyAnalysisPlot1(F,magnitudeY,phaseY,NFFT);
legend([ muscle ':Emg (RAW) Deaveraged']);
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_bode_emg_deavg_' num2str(file_counter)]);
y = emg_filt_freq_spectrum_raw;
NFFT = length(y);
Y = fft(y,NFFT);
F = ((0:1/NFFT:1-1/NFFT)*Fs).';
magnitudeY = abs(Y);        % Magnitude of the FFT
phaseY = unwrap(angle(Y));  % Phase of the FFT
helperFrequencyAnalysisPlot1(F,magnitudeY,phaseY,NFFT);
legend([ muscle ':Emg (RAW) Bandpassed']);
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_raw/' muscle '_bode_emg_filt_' num2str(file_counter)]);


%%%%%%%%%%%%%%%%%%%%%%%%%%% HP EMG
y = emg_deavg_hp;
NFFT = length(y);
Y = fft(y,NFFT);
F = ((0:1/NFFT:1-1/NFFT)*Fs).';
magnitudeY = abs(Y);        % Magnitude of the FFT
phaseY = unwrap(angle(Y));  % Phase of the FFT
helperFrequencyAnalysisPlot1(F,magnitudeY,phaseY,NFFT);
legend([ muscle ':Emg (HP) Deaveraged']);
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_bode_emg_deavg_' num2str(file_counter)]);
y = emg_filt_freq_spectrum_hp;
NFFT = length(y);
Y = fft(y,NFFT);
F = ((0:1/NFFT:1-1/NFFT)*Fs).';
magnitudeY = abs(Y);        % Magnitude of the FFT
phaseY = unwrap(angle(Y));  % Phase of the FFT
helperFrequencyAnalysisPlot1(F,magnitudeY,phaseY,NFFT);
legend([ muscle ':Emg (HP) Bandpassed']);
savefig(['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/plots/emg_hp/' muscle '_bode_emg_filt_' num2str(file_counter)]);

%%
%%
%%
%%
%%
%%
%%
%%
%%
%%
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NON_ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%High Pass Filter by zeroing and inverse fft 
Ylp = Y;
fc = 250;
Ylp(F>=fc & F<=Fs-fc) = 0;
helperFrequencyAnalysisPlot1(F,abs(Ylp),unwrap(angle(Ylp)),NFFT,...
  'Frequency components above 1 kHz have been zeroed')
ylp = ifft(Ylp,'symmetric');
figure;
plot(time_samples,emg_data);
hold on;
plot(time_samples,ylp);

%% Fourier Transform prototyping - Type 1
fftLength = 1024;
fs = sample_rate;
x = emg_filt_freq_spectrum;
sigLength = length(x);
win = rectwin(sigLength);
y = fft(x.*win,fftLength);
figLength = fftLength/2 + 1;
% Plot the Magnitude Response in Linear Scale
subplot(2,1,1);
plot([1:figLength]*fs/(2*figLength),abs(y(1:figLength)));
subplot(2,1,2);
% Plot the Magnitude Response in Log Scale
plot([1:figLength]*fs/(2*figLength),(20*log10(abs(y(1:figLength)))));
%% Analyzing linear trends between torque and emg with varying cutt-off frequencies
%Filtering EMG data - butterworthmagnitudeY = abs(Y);        % Magnitude of the FFT
phaseY = unwrap(angle(Y));  % Phase of the FFT

helperFrequencyAnalysisPlot1(F,magnitudeY,phaseY,NFFT)
%close all;
% Enter voltage at no contraction here
voltage_zero_contraction = mean(emg_data);
% full wave rectification
emg_rect_data = zeros(length(emg_data),1);
for i=1:length(emg_data)
    emg_rect_data(i) = abs(emg_data(i) - voltage_zero_contraction);
end
% Butterworth filter
wn = sample_rate/2; % nyquist frequency for digital data - max possible freq of data
windowSize = 10;
a2 = 1;
b2 = (1/windowSize)*ones(1,windowSize);
torque_filt_data = filter(b2,a2,torque_data);
for varFreq = 1:200
    cuttoff_freq = varFreq %% cutt off freq for low pass
    norm_freq = cuttoff_freq/wn; % [rads/sample]
    [a b]=butter(2,norm_freq); %create butterworth coefficients 
    emg_filt_envelope = filtfilt(a,b,emg_rect_data); 

    % Analysis of trends between emg and torque
    mdl = fitlm(emg_filt_envelope,torque_filt_data);
    Rvalues(varFreq) = mdl.Rsquared.Ordinary;
end
freq = 1:200;
figure();
scatter(freq,Rvalues);

%% test
voltage_zero_contraction =  mean(emg_data);
% full wave rectification
emg_rect_data = zeros(length(emg_data),1);
for i=1:length(emg_data)
    emg_rect_data(i) = emg_data(i) - voltage_zero_contraction;
end
%% gSUSB Amp only EMG test
read_buffer = load('g_emg_raw_3', '-mat');
gamp_emg_raw = read_buffer.y;
gamp_emg_raw = gamp_emg_raw';

clear read_buffer;
read_buffer = load('g_emg_filtered_3', '-mat');
gamp_emg_filtered = read_buffer.y;
gamp_emg_filtered = gamp_emg_filtered';

figure;
plot(gamp_emg_raw(:,1),gamp_emg_raw(:,2));
hold on;
plot(gamp_emg_filtered(:,1),gamp_emg_filtered(:,2));
hold off;
%%






%%%%%%%%%%%%% FOR CSV FILES AND ALIGNED %%%%%%%%%%%%
%
%
%   
%
%
%
%% Routine #1 Load files
% Set the file index to be read; file index is appeneded to the file name
clear all;
close all;
sample_rate = 4800;
live_durationn = 100;
live_duration = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_counter = 10;
curr_angle = 60;
routine = 'R3_SLOPE';
file_name = 'slope_contraction_';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % file names for torque
file_name_torque = ['data/angle_' num2str(curr_angle) '/' routine '/torque/' file_name num2str(file_counter) '.csv'];
% filenames for VMO, QF1, QF2, VL
file_name_VMO_raw_gemg = ['data/angle_' num2str(curr_angle) '/' routine '/emg/VMO_raw_emg_' file_name num2str(file_counter) '.csv'];
file_name_VMO_hp_gemg = ['data/angle_' num2str(curr_angle) '/' routine '/emg/VMO_hp_emg_' file_name num2str(file_counter) '.csv'];
file_name_QF1_raw_gemg = ['data/angle_' num2str(curr_angle) '/' routine '/emg/QF1_raw_emg_' file_name num2str(file_counter) '.csv'];
file_name_QF1_hp_gemg = ['data/angle_' num2str(curr_angle) '/' routine '/emg/QF1_hp_emg_' file_name num2str(file_counter) '.csv'];
file_name_QF2_raw_gemg = ['data/angle_' num2str(curr_angle) '/' routine '/emg/QF2_raw_emg_' file_name num2str(file_counter) '.csv'];
file_name_QF2_hp_gemg = ['data/angle_' num2str(curr_angle) '/' routine '/emg/QF2_hp_emg_' file_name num2str(file_counter) '.csv'];
file_name_VL_raw_gemg = ['data/angle_' num2str(curr_angle) '/' routine '/emg/VL_raw_emg_' file_name num2str(file_counter) '.csv'];
file_name_VL_hp_gemg = ['data/angle_' num2str(curr_angle) '/' routine '/emg/VL_hp_emg_' file_name num2str(file_counter) '.csv'];

%Read all data
torque_raw = csvread(file_name_torque);
VMO_raw_gemg = csvread(file_name_VMO_raw_gemg);
VMO_hp_gemg = csvread(file_name_VMO_hp_gemg);
QF1_raw_gemg = csvread(file_name_QF1_raw_gemg);
QF1_hp_gemg = csvread(file_name_QF1_hp_gemg);
QF2_raw_gemg = csvread(file_name_QF2_raw_gemg);
QF2_hp_gemg = csvread(file_name_QF2_hp_gemg);
VL_raw_gemg = csvread(file_name_VL_raw_gemg);
VL_hp_gemg = csvread(file_name_VL_hp_gemg);

% Split into data and time samples
time_samples = torque_raw(:,1);
torque_data = torque_raw(:,2);
vmo_emg_raw = VMO_raw_gemg(:,2);
vmo_emg_data = VMO_hp_gemg(:,2);
qf1_emg_raw = QF1_raw_gemg(:,2);
qf1_emg_data = QF1_hp_gemg(:,2);
qf2_emg_raw = QF2_raw_gemg(:,2);
qf2_emg_data = QF2_hp_gemg(:,2);
vl_emg_raw = VL_raw_gemg(:,2);
vl_emg_data = VL_hp_gemg(:,2);

% PLOT DATA TO VERIFY READ
close all;

% VMO
figure();
subplot(3,1,1);
grid on;
plot(time_samples,vmo_emg_raw);
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'));
title('VMO: Raw EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,2);
plot(time_samples,vmo_emg_data);
ylim([-5000 5000]);
title('VMO: High Passed (18Hz) EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,3);
plot(time_samples,torque_data);
ylim([-500 500]);
title('Raw Force of desired time period');
xlabel('Time[s]');
ylabel('Force[N]');

% QF1
figure();
subplot(3,1,1);
grid on;
plot(time_samples,qf1_emg_raw);
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'));
title('QF1: Raw EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,2);
plot(time_samples,qf1_emg_data);
ylim([-5000 5000]);
title('QF1: High Passed (18Hz) EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,3);
plot(time_samples,torque_data);
ylim([-500 500]);
title('Raw Force of desired time period');
xlabel('Time[s]');
ylabel('Force[N]');

% QF2
figure();
subplot(3,1,1);
grid on;
plot(time_samples,qf2_emg_raw);
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'));
title('QF2: Raw EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,2);
plot(time_samples,qf2_emg_data);
ylim([-5000 5000]);
title('QF2: High Passed (18Hz) EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[mV]');
subplot(3,1,3);
plot(time_samples,torque_data);
ylim([-500 500]);
title('Raw Force of desired time period');
xlabel('Time[s]');
ylabel('Force[N]');

% VL
figure();
grid on;
subplot(3,1,1);
plot(time_samples,vl_emg_raw);
set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'))
title('VL: Raw EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[V]');
subplot(3,1,2);
plot(time_samples,vl_emg_data);
ylim([-5000 5000]);
title('VL: High Passed (18Hz) EMG of desired time period');
xlabel('Time[s]');
ylabel('Amplitude[V]');
subplot(3,1,3);
plot(time_samples,torque_data);
ylim([-500 500]);
title('Raw Force of desired time period');
xlabel('Time[s]');
ylabel('Force[N]');

%% Save data for Sunaal
file_counter = 1;
participant_name = 'Raghav';
file_name_save = ['data/' participant_name '/angle_' num2str(curr_angle) '/' routine '/' file_name num2str(file_counter) '.mat'];
save(file_name_save, 'torque_data', ...
                   'time_samples', ...
                   'live_duration', ...
                   'sample_rate', ...                   
                   'vmo_emg_raw', ...
                   'vmo_emg_data', ...
                   'qf1_emg_raw', ...
                   'qf1_emg_data', ...               
                   'qf2_emg_raw', ...
                   'qf2_emg_data', ...                  
                   'vl_emg_raw', ...
                   'vl_emg_data' ... 
    );