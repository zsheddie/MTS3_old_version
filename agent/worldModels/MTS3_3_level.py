import torch
import sys
sys.path.append('E:\\KIT/毕设\\世界模型\\MTS3代码注释随意修改\\MTS3_release-master\\MTS3_release-master')

from agent.worldModels.SensorEncoders.propEncoder import Encoder
from agent.worldModels.gaussianTransformations.gaussian_marginalization import Predict
from agent.worldModels.gaussianTransformations.gaussian_conditioning import Update
from agent.worldModels.Decoders.propDecoder import SplitDiagGaussianDecoder

nn = torch.nn
'''
main structure MTS3
class MTS3():
    def __init__(self, input_shape=None, action_dim=None, config=None, use_cuda_if_available: bool = True):
    def _intialize_mean_covar(self, batch_size, diagonal=False, scale=1.0, learn=False):
    def _create_time_embedding(self, batch_size, time_steps):
    def _pack_variances(self, variances):
    def _unpack_variances(self, variances):
    def forward(self, obs_seqs, action_seqs, obs_valid_seqs):   ## 输入：观测序列，动作序列，可用的观测序列
        for k in range(0, obs_seqs.shape[1], self.H):
        for k in range(0,num_episodes): 
            for t in range(current_episode_len):
        decode
        return
'''

'''
Tip: in config self.ltd = lod 
in context_predict ltd is doubled
in task_predict ltd is considered lod and lsd is doubled inside
提示：在配置中 self.ltd = lod
在 context_predict ltd 中翻倍
在task_predict ltd中被认为是lod,lsd在里面被加倍
'''
class MTS3_3_level(nn.Module):
    """
    MTS3 model
    Inference happen in such a way that first episode is used for getting an intial task posterioer and then the rest of the episodes are used for prediction by the worker
    Maybe redo this logic based on original implementation or use a different method that helps control too ??
    """

    def __init__(self, input_shape=None, action_dim=None, config=None, use_cuda_if_available: bool = True):
        """
        @param obs_dim: dimension of observations to train on  
        @param action_dim: dimension of control signals
        @param inp_shape: shape of the input observations
        @param config: config dict
        @param dtype:
        @param use_cuda_if_available:
        """
        super(MTS3_3_level, self).__init__()
        if config == None:
            raise ValueError("config cannot be None, pass an omegaConf File")
        else:
            self.c = config
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._obs_shape = input_shape
        self._action_dim = action_dim
        self._lod = self.c.mts3.latent_obs_dim   ## latent observation dimension
        self._lsd = 2*self._lod                  ## latent state dimension 为啥是lod 2倍
        self.H =self.c.mts3.time_scale_multiplier   ## 时间尺度因子
        self.H_3 =self.c.mts3.time_scale_multiplier_3
        self._time_embed_dim = self.c.mts3.manager.abstract_obs_encoder.time_embed.dim    
        assert self._time_embed_dim == self.c.mts3.manager.abstract_act_encoder.time_embed.dim, \
                                            "Time Embedding Dimensions for obs and act encoder should be same"   ## 实际就是经过encoder后的高维向量特征
        self._pixel_obs = self.c.mts3.pixel_obs   ## 这是啥为啥有pixel根本没用到啊
        self._decode_reward = self.c.mts3.decode.reward   ## 哪来的reward？
        self._decode_obs = self.c.mts3.decode.obs   ## 这又是啥



        ### Define the encoder and decoder
        obsEnc = Encoder(self._obs_shape[-1], self._lod, self.c.mts3.worker.obs_encoder) ## TODO: config
        self._obsEnc = obsEnc.to(self._device)   ## 将encoder运算推到gpu

        absObsEnc = Encoder(self._obs_shape[-1] + self._time_embed_dim, self._lod, self.c.mts3.manager.abstract_obs_encoder) ## TODO: config   ## abstract observation encoder
        self._absObsEnc = absObsEnc.to(self._device)

        absActEnc = Encoder(self._action_dim + self._time_embed_dim, self._lsd, self.c.mts3.manager.abstract_act_encoder) ## TODO: config
        self._absActEnc = absActEnc.to(self._device)

        obsDec = SplitDiagGaussianDecoder(latent_obs_dim=self._lod, out_dim=self._obs_shape[-1], config=self.c.mts3.worker.obs_decoder) ## TODO: config
        self._obsDec = obsDec.to(self._device)

        if self._decode_reward:     ## 如果有reward的话
            rewardDec = SplitDiagGaussianDecoder(latent_obs_dim=self._lod, out_dim=1, config=self.c.mts3.worker.reward_decoder) ## TODO: config   ## 这是个直接引入的函数
            self._rewardDec = rewardDec.to(self._device)


        self._task_predict = Predict(latent_obs_dim=self._lod, act_dim=self._action_dim, hierarchy_type = "submanager", config=self.c.mts3.worker).to(self._device)
        ### Define the gaussian layers for both levels
        self._state_predict = Predict(latent_obs_dim=self._lod, act_dim=self._action_dim, hierarchy_type = "worker", config=self.c.mts3.worker).to(self._device)    ## 微时间尺度下的是worker
                                                                                    ## initiate worker marginalization layer for state prediction
        ## predict 是从 gaussian_marginalization 中导入的
        self._proj_predict = Predict(latent_obs_dim=self._lod, act_dim=self._action_dim, hierarchy_type = "manager", config=self.c.mts3.manager).to(self._device)   ## 大时间尺度下的是manager
                                                                                    ## initiate manager marginalization layer for task prediction
        ## update 是从 gaussian_conditioning 中导入的
        self._obsUpdate = Update(latent_obs_dim=self._lod, memory = True, config = self.c).to(self._device) ## memory is true   这里是指有没有acRKN中的memory机制吗
        self._taskUpdate = Update(latent_obs_dim=self._lod, memory = True, config = self.c).to(self._device) ## memory is true

        self._action_Infer = Update(latent_obs_dim=self._lsd, memory = False, config = self.c).to(self._device) ## memory is false

    def _intialize_mean_covar(self, batch_size, diagonal=False, scale=1.0, learn=False):   ## 初始化均值、方差
        if learn:
            pass
        else:
            if diagonal:  ## 是对角side为0，否则为1
                init_state_covar_ul = scale * torch.ones(batch_size, self._lsd)    ## ul指prior covariance matrix 中的对角项upper、lower

                initial_mean = torch.zeros(batch_size, self._lsd).to(self._device)    
                icu = init_state_covar_ul[:, :self._lod].to(self._device)       ## initial covariance upper
                icl = init_state_covar_ul[:, self._lod:].to(self._device)       ## initial covariance lower
                ics = torch.zeros(1, self._lod).to(self._device)  ### side covariance is zero  ##initial covariance side

                initial_cov = [icu, icl, ics]
            else:
                init_state_covar_ul = scale * torch.ones(batch_size, self._lsd)

                initial_mean = torch.zeros(batch_size, self._lsd).to(self._device)
                icu = init_state_covar_ul[:, :self._lod].to(self._device)     ## [:x,]表示只读取某一维度的前x项
                icl = init_state_covar_ul[:, self._lod:].to(self._device)
                ics = torch.ones(1, self._lod).to(self._device)  ### side covariance is one

                initial_cov = [icu, icl, ics]
        return initial_mean, initial_cov


    def _create_time_embedding(self, batch_size, time_steps):
        """
        Creates a time embedding for the given batch size and time steps    创建时间嵌入
        of the form (batch_size, time_steps, 1)"""
        time_embedding = torch.zeros(batch_size, time_steps, 1).to(self._device)
        for i in range(time_steps):
            time_embedding[:, i, :] = i / time_steps     ## 表示成每个step占总时间的百分比
        return time_embedding
    
    def _pack_variances(self, variances):
        """
        pack list of variances (upper, lower, side) into a single tensor
        """
        return torch.cat(variances, dim=-1)  ## 将2个张量拼接在一起
    
    def _unpack_variances(self, variances):
        """
        unpack list of variances (upper, lower, side) from a single tensor
        """
        return torch.split(variances, self._lod, dim=-1)
    
    def forward(self, obs_seqs, action_seqs, obs_valid_seqs):   ## 输入：观测序列，动作序列，可用的观测序列
        '''
        obs_seqs: sequences of timeseries of observations (batch x time x obs_dim)   观察的时间序列
        action_seqs: sequences of timeseries of actions (batch x time x obs_dim)     动作的时间序列
        '''
        ##################################### (Super) Manager ############################################
        # prepare list for return
        prior_proj_mean_list = []    ## 创建用于储存prior tesk mean的列表
        prior_proj_cov_list = []     

        post_proj_mean_list = []    ## 记录所有后验的 task_mean 的集合
        post_proj_cov_list = []

        abs_act_list = []

        ### initialize mean and covariance for the first time step
        proj_prior_mean_init, proj_prior_cov_init = self._intialize_mean_covar(obs_seqs.shape[0], scale=self.c.mts3.manager.initial_state_covar, learn=False)
        main_skill_prior_mean, main_skill_prior_cov = self._intialize_mean_covar(obs_seqs.shape[0], diagonal=True,  learn=False)
        H_super = self.H*self.H_3
        ### loop over individual episodes in steps of H (Coarse粗 time scale / manager)
        for k in range(0, obs_seqs.shape[1], H_super):   ## obs 的dim=0 是batch吗？？
            district_num = int(k // H_super)   ## k对H整除取结果，表示当前是第几个时间窗口
            if k==0:
                proj_prior_mean = proj_prior_mean_init
                proj_prior_cov = proj_prior_cov_init
            ### encode the observation set with time embedding to get abstract observation     ## 将observation set和time embedding送入编码器来获得抽象观察
            current_obs_seqs = obs_seqs[:, k:k+H_super, :]      ## 当前时间窗口的obs集合
            time_embedding = self._create_time_embedding(current_obs_seqs.shape[0], current_obs_seqs.shape[1])   ## 格式[batch_size, time_steps]
                                                                        # [x] made sure works with episodes < H
            beta_n_mean, beta_n_var = self._absObsEnc(torch.cat([current_obs_seqs, time_embedding], dim=-1))    ## 通过encoder将当前obs集合抽象化

            ### get task valid for the current episode
            obs_valid = obs_valid_seqs[:, k:k+H_super, :] 
                        #[x] created in learn class, with interwindow (whole windows masked) and intrawindow masking
            ### update the task posterior with beta_current
            proj_post_mean, proj_post_cov = self._taskUpdate(proj_prior_mean, proj_prior_cov, beta_n_mean, beta_n_var, obs_valid)

            ### infer the abstract action with bayesian aggregation
            current_act_seqs = action_seqs[:, k+H_super: k+2*H_super, :] 
            
            if current_act_seqs.shape[1] != 0:             ## 如果序列长度不为0
                ## skip prior calculation for the next window if there is no action in the current window    ## 若当前时间窗口没有动作则跳过下一时间窗口的先验计算（因为没有任何动作，所以先验不会被更新）
                
                alpha_n_mean, alpha_n_var = self._absActEnc(torch.cat([current_act_seqs, time_embedding], dim=-1)) \
                                            ## encode the action set with time embedding
                abs_act_mean, abs_act_var = self._action_Infer(main_skill_prior_mean, main_skill_prior_cov, alpha_n_mean, \
                                                                alpha_n_var, None) ##BA with all actions valid                 ## action inference就是贝叶斯聚合操作


                ### predict the next task mean and covariance using manager marginalization layer
                ### with the current task posterior and abstract action as causal factors                ## 用高斯marginalization更新先验（通过当前后验和抽象动作集合）
                mean_list_causal_factors = [proj_post_mean, abs_act_mean]
                cov_list_causal_factors = [proj_post_cov, abs_act_var]
                proj_next_mean, proj_next_cov = self._proj_predict(mean_list_causal_factors, cov_list_causal_factors) #[.]: absact inference some problem fixed.
                
                ### update the task prior
                proj_prior_mean, proj_prior_cov = proj_next_mean, proj_next_cov   ## 更新prior mean 和var 从initial到当前数值

                ### append the task mean and covariance to the list
                prior_proj_mean_list.append(proj_prior_mean)   ##这里的proj_prior_mean已经是一个torch张量，而list列表中包含n个这样的张量
                prior_proj_cov_list.append(self._pack_variances(proj_prior_cov)) ## append the packed covariances
                abs_act_list.append(abs_act_mean)

            
            post_proj_mean_list.append(proj_post_mean)
            post_proj_cov_list.append(self._pack_variances(proj_post_cov)) ## append the packed covariances


        ### stack the list to get the final tensors  堆叠列表以获得最终张量
        prior_proj_means = torch.stack(prior_proj_mean_list, dim=1)    ## stack是生成新维度的拼接，但生成的新维度是第二个维度dim=1，而不是最后一个维度
        prior_proj_covs = torch.stack(prior_proj_cov_list, dim=1)
        post_proj_means = torch.stack(post_proj_mean_list, dim=1)
        post_proj_covs = torch.stack(post_proj_cov_list, dim=1)
        abs_acts = torch.stack(abs_act_list, dim=1)

        ### get the number of districts from the length of post_task_mean
        num_district = post_proj_means.shape[1]    ## 有多少个post_task_mean说明当前有多少时间窗口
        

        ##################################### (sub)Manager ############################################
        # prepare list for return
        global_prior_task_mean_list = []    ## 创建用于储存prior task mean的列表
        global_prior_task_cov_list = []     

        global_post_task_mean_list = []    ## 记录所有后验的 task_mean 的集合
        global_post_task_cov_list = []
        num_episodes = []
        sub_abs_act_list = []

        ### initialize mean and covariance for the first time step
        task_prior_mean_init, task_prior_cov_init = self._intialize_mean_covar(obs_seqs.shape[0], scale=self.c.mts3.manager.initial_state_covar, learn=False)
        skill_prior_mean, skill_prior_cov = self._intialize_mean_covar(obs_seqs.shape[0], diagonal=True,  learn=False)

        for n in range(0,num_district):
            if n == 0:
                task_prior_mean = task_prior_mean_init
                task_prior_cov = task_prior_cov_init

            prior_task_mean_list = []
            prior_task_cov_list = []
            post_task_mean_list = []
            post_task_cov_list = []

            proj_mean = post_proj_means[:, n, :]
            proj_cov = self._unpack_variances(post_proj_covs[:, n, :])

            current_obs_seqs = obs_seqs[:, n*H_super: (n+1)*H_super, :]
            current_obs_valid_seqs = obs_valid_seqs[:, n*H_super: (n+1)*H_super, :]
            current_act_seqs = action_seqs[:, n*H_super:(n+1)*H_super, :]
            current_district_len = current_obs_seqs.shape[1]

            for k in range(0, current_district_len, self.H):
                current_obs_seq = current_obs_seqs[:, k:k+self.H, :]
                time_embedding = self._create_time_embedding(current_obs_seq.shape[0], current_obs_seq.shape[1])
                beta_k_mean, beta_k_var = self._absObsEnc(torch.cat([current_obs_seq, time_embedding], dim=-1))

                current_obs_valid = current_obs_valid_seqs[:, k:k+self.H, :]
                task_post_mean, task_post_cov = self._taskUpdate(task_prior_mean, task_prior_cov, beta_k_mean, beta_k_var, current_obs_valid)
                ### ^posterior udate^ ###

                ### predict the next state mean and covariance using the marginalization layer for submanager
                current_act_seq = current_act_seqs[:, k:k+self.H, :]
                
                if current_act_seq.shape[1] != 0:             ## 如果序列长度不为0
                ## skip prior calculation for the next window if there is no action in the current window    ## 若当前时间窗口没有动作则跳过下一时间窗口的先验计算（因为没有任何动作，所以先验不会被更新）
                    
                    alpha_k_mean, alpha_k_var = self._absActEnc(torch.cat([current_act_seq, time_embedding], dim=-1)) \
                                            ## encode the action set with time embedding
                    abs_act_mean, abs_act_var = self._action_Infer(skill_prior_mean, skill_prior_cov, alpha_k_mean, \
                                                                alpha_k_var, None) ##BA with all actions valid                 ## action inference就是贝叶斯聚合操作


                    ### predict the next task mean and covariance using manager marginalization layer
                    ### with the current task posterior and abstract action as causal factors                ## 用高斯marginalization更新先验（通过当前后验和抽象动作集合）
                    mean_list_causal_factors = [task_post_mean, abs_act_mean, proj_mean]
                    cov_list_causal_factors = [task_post_cov, abs_act_var, proj_cov]
                    #print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                    #print(task_post_mean.shape, abs_act_mean.shape, proj_mean.shape)
                    #print(task_post_cov[0].shape, task_post_cov[1].shape, task_post_cov[2].shape, abs_act_var[0].shape, abs_act_var[1].shape, abs_act_var[2].shape)
                    task_next_mean, task_next_cov = self._task_predict(mean_list_causal_factors, cov_list_causal_factors) #[.]: absact inference some problem fixed.

                    ### update the task prior
                    task_prior_mean, task_prior_cov = task_next_mean, task_next_cov   ## 更新prior mean 和var 从initial到当前数值

                    ### append the task mean and covariance to the list
                    prior_task_mean_list.append(task_prior_mean)
                    prior_task_cov_list.append(self._pack_variances(task_prior_cov)) ## append the packed covariances
                    sub_abs_act_list.append(abs_act_mean)

                post_task_mean_list.append(task_post_mean)
                post_task_cov_list.append(self._pack_variances(task_post_cov)) ## append the packed covariances

            ### stack the list to get the final tensors  堆叠列表以获得最终张量
            prior_task_means = torch.stack(prior_task_mean_list, dim=1)
            prior_task_covs = torch.stack(prior_task_cov_list, dim=1)
            post_task_means = torch.stack(post_task_mean_list, dim=1)
            post_task_covs = torch.stack(post_task_cov_list, dim=1)
            sub_abs_acts = torch.stack(sub_abs_act_list, dim=1)   ## useful?

            global_prior_task_mean_list.append(prior_task_means)
            global_prior_task_cov_list.append(prior_task_covs)  
            global_post_task_mean_list.append(post_task_means)
            global_post_task_cov_list.append(post_task_covs)

        global_prior_task_mean = torch.cat(global_prior_task_mean_list, dim = 1)
        global_prior_task_cov = torch.cat(global_prior_task_cov_list, dim = 1)
        global_post_task_mean = torch.cat(global_post_task_mean_list, dim = 1)
        global_post_task_cov = torch.cat(global_post_task_cov_list, dim = 1)
        

        '''
            if n == 0: 
                prior_task_means = prior_task_means_new
                prior_task_covs = prior_task_covs_new
                post_task_means = post_task_means_new
                print('n=0时, post_task_means.shape=', post_task_means.shape)

                post_task_covs = post_task_covs_new
                abs_acts = abs_acts_new
            else:
                prior_task_means = torch.cat((prior_task_means,prior_task_means_new), dim = 1)
                prior_task_covs = torch.cat((prior_task_covs,prior_task_covs_new), dim = 1)
                post_task_means = torch.cat((post_task_means,post_task_means_new), dim = 1)
                print('n=', n, '时, post_task_means.shape=', post_task_means.shape)
                post_task_covs = torch.cat((post_task_covs,post_task_covs_new), dim = 1)
                abs_acts = torch.cat((abs_acts,abs_acts_new), dim = 1)
            ### get the number of episodes from the length of post_task_mean
            #print(post_task_means.shape)
            #num_episodes[n] = post_task_means_new.shape[1]    ## 有多少个post_task_mean说明当前有多少时间窗口   ## seems not necessary, might delete later
        '''

        """
        ### loop over individual episodes in steps of H (Coarse粗 time scale / manager)
        for k in range(0, obs_seqs.shape[1], self.H):   ## obs 的dim=0 是batch吗？？  是的
            episode_num = int(k // self.H)   ## k对H整除取结果,表示当前是第几个时间窗口
            if k==0:
                task_prior_mean = task_prior_mean_init
                task_prior_cov = task_prior_cov_init
            ### encode the observation set with time embedding to get abstract observation     ## 将observation set和time embedding送入编码器来获得抽象观察
            current_obs_seqs = obs_seqs[:, k:k+self.H, :]      ## 当前时间窗口的obs集合
            time_embedding = self._create_time_embedding(current_obs_seqs.shape[0], current_obs_seqs.shape[1])   ## 格式[batch_size, time_steps]
                                                                        # [x] made sure works with episodes < H
            beta_k_mean, beta_k_var = self._absObsEnc(torch.cat([current_obs_seqs, time_embedding], dim=-1))    ## 通过encoder将当前obs集合抽象化

            ### get task valid for the current episode
            obs_valid = obs_valid_seqs[:, k:k+self.H, :] 
                        #[x] created in learn class, with interwindow (whole windows masked) and intrawindow masking
            ### update the task posterior with beta_current
            task_post_mean, task_post_cov = self._taskUpdate(task_prior_mean, task_prior_cov, beta_k_mean, beta_k_var, obs_valid)
            ##################### 到这里为止是更新慢时间尺度的task

            ### infer the abstract action with bayesian aggregation
            current_act_seqs = action_seqs[:, k+self.H: k+2*self.H, :]    ## 当前时间窗口的动作序列为什么是k+H到K+2H？
            if current_act_seqs.shape[1] != 0:             ## 如果序列长度不为0
                ## skip prior calculation for the next window if there is no action in the current window    ## 若当前时间窗口没有动作则跳过下一时间窗口的先验计算（因为没有任何动作，所以先验不会被更新）
                alpha_k_mean, alpha_k_var = self._absActEnc(torch.cat([current_act_seqs, time_embedding], dim=-1)) \
                                            ## encode the action set with time embedding
                abs_act_mean, abs_act_var = self._action_Infer(skill_prior_mean, skill_prior_cov, alpha_k_mean, \
                                                                alpha_k_var, None) ##BA with all actions valid                 ## action inference就是贝叶斯聚合操作


                ### predict the next task mean and covariance using manager marginalization layer
                ### with the current task posterior and abstract action as causal factors                ## 用高斯marginalization更新先验（通过当前后验和抽象动作集合）
                mean_list_causal_factors = [task_post_mean, abs_act_mean, proj_mean]
                cov_list_causal_factors = [task_post_cov, abs_act_var]
                #task_next_mean, task_next_cov = self._task_predict(mean_list_causal_factors, cov_list_causal_factors) #[.]: absact inference some problem fixed.
                task_next_mean, task_next_cov = self._state_predict(mean_list_causal_factors, cov_list_causal_factors)

                ### update the task prior
                task_prior_mean, task_prior_cov = task_next_mean, task_next_cov   ## 更新prior mean 和var 从initial到当前数值

                ### append the task mean and covariance to the list
                prior_task_mean_list.append(task_prior_mean)
                prior_task_cov_list.append(self._pack_variances(task_prior_cov)) ## append the packed covariances
                abs_act_list.append(abs_act_mean)

            post_task_mean_list.append(task_post_mean)
            post_task_cov_list.append(self._pack_variances(task_post_cov)) ## append the packed covariances

            

        ### stack the list to get the final tensors  堆叠列表以获得最终张量
        prior_task_means = torch.stack(prior_task_mean_list, dim=1)
        prior_task_covs = torch.stack(prior_task_cov_list, dim=1)
        post_task_means = torch.stack(post_task_mean_list, dim=1)
        post_task_covs = torch.stack(post_task_cov_list, dim=1)
        abs_acts = torch.stack(abs_act_list, dim=1)

        ### get the number of episodes from the length of post_task_mean
        num_episodes = post_task_means.shape[1]    ## 有多少个post_task_mean说明当前有多少时间窗口
        """
        
        ##################################### Worker ############################################
        ### using the task prior, predict the observation mean and covariance for fine time scale / worker
        ### create a meta_list of prior and posterior states

        global_state_prior_mean_list = []
        global_state_prior_cov_list = []
        global_state_post_mean_list = []
        global_state_post_cov_list = []

        state_prior_mean_init, state_prior_cov_init = self._intialize_mean_covar(obs_seqs.shape[0],scale=self.c.mts3.worker.initial_state_covar, learn=False)

        for n in range(0, num_district):
            print(n, "just to let u know I'm still running...")
            #torch.cuda.empty_cache()
            if n == 0:
                state_prior_mean = state_prior_mean_init
                state_prior_cov = state_prior_cov_init
            ### create list of state mean and covariance 
            

            #proj_mean = post_proj_means[:, n, :]   ## not necessary?
            #proj_cov = self._unpack_variances(post_proj_covs[:, n, :])
            task_means = global_post_task_mean[:, n*self.H_3:(n+1)*self.H_3, :]
            task_covs = global_post_task_cov[:, n*self.H_3:(n+1)*self.H_3]
            ### get the obs, action for the current episode
            current_obs_seqs = obs_seqs[:, n*H_super:(n+1)*H_super, :]
            current_act_seqs = action_seqs[:, n*H_super:(n+1)*H_super, :]
            current_obs_valid_seqs = obs_valid_seqs[:, n*H_super:(n+1)*H_super, :]
            current_district_len = current_obs_seqs.shape[1]

            for k in range(int(current_district_len//self.H)):
                #torch.cuda.empty_cache()
                #k = int(w // self.H)
                prior_state_mean_list = []
                prior_state_cov_list = []
                post_state_mean_list = []
                post_state_cov_list = []

                task_mean = task_means[:, k, :]
                task_cov = self._unpack_variances(task_covs[:, k, :])

                ### get the obs, action for the current episode
                current_obs_seq = obs_seqs[:, k*self.H:(k+1)*self.H, :]
                current_act_seq = action_seqs[:, k*self.H:(k+1)*self.H, :]
                current_obs_valid_seq = obs_valid_seqs[:, k*self.H:(k+1)*self.H, :]
                current_episode_len = current_obs_seq.shape[1]

                for t in range(current_episode_len): # [x] made sure works with episodes < H
                    #print(n,k,t)
                    #torch.cuda.empty_cache()
                    ### encode the observation (no time embedding)
                    current_obs = current_obs_seq[:, t, :]

                    obs_mean, obs_var = self._obsEnc(current_obs)
                    ## expand dims to make it compatible with the update step (which expects a 3D tensor)
                    obs_mean = torch.unsqueeze(obs_mean, dim=1)
                    obs_var = torch.unsqueeze(obs_var, dim=1)

                    ### update the state posterior
                    current_obs_valid = current_obs_valid_seq[:, t, :]
                    ## expand dims to make it compatible with the update step (which expects a 3D tensor)
                    current_obs_valid = torch.unsqueeze(current_obs_valid, dim=1)
                    state_post_mean, state_post_cov = self._obsUpdate(state_prior_mean, state_prior_cov, obs_mean, obs_var, current_obs_valid)

                    ### predict the next state mean and covariance using the marginalization layer for worker
                    current_act = current_act_seq[:, t, :]
                    mean_list_causal_factors = [state_post_mean, current_act, task_mean]
                    cov_list_causal_factors = [state_post_cov, task_cov]
                
                    state_next_mean, state_next_cov = self._state_predict(mean_list_causal_factors, cov_list_causal_factors)

                    ### update the state prior
                    state_prior_mean, state_prior_cov = state_next_mean, state_next_cov  ### this step also makes sure every episode 
                                                                                        ### starts with the prior of the previous episode
                    ## 为啥不直接 state_prior_mean, state_prior_cov = self._state_predict(mean_list_causal_factors, cov_list_causal_factors) ？

                    ### concat 
                    ### append the state mean and covariance to the list
                    prior_state_mean_list.append(state_prior_mean)    ##最后格式应该是list中有30个[20,60]的tensor
                    prior_state_cov_list.append(torch.cat(state_prior_cov, dim=-1))
                    post_state_mean_list.append(state_post_mean)
                    post_state_cov_list.append(torch.cat(state_post_cov, dim=-1))

                ## detach the state prior mean and covariance to make sure the next episode starts with the prior of the previous episode  将状态先验均值和协方差分离，以确保下一窗口从上一窗口的先验开始
                state_prior_mean = state_prior_mean.detach()
                state_prior_cov = [cov.detach() for cov in state_prior_cov]

                ### stack the list to get the final tensors   将list堆叠张量
                prior_state_means = torch.stack(prior_state_mean_list, dim=1)    ##tensor.size = [2, 150, 60]
                prior_state_covs = torch.stack(prior_state_cov_list, dim=1)
                post_state_means = torch.stack(post_state_mean_list, dim=1)
                post_state_covs = torch.stack(post_state_cov_list, dim=1)

                ### append the state mean and covariance to the list
                global_state_prior_mean_list.append(prior_state_means)   
                global_state_prior_cov_list.append(prior_state_covs)
                global_state_post_mean_list.append(post_state_means)
                global_state_post_cov_list.append(post_state_covs)

        ### concat along the episode dimension
        global_state_prior_means = torch.cat(global_state_prior_mean_list, dim=1)
        global_state_prior_covs = torch.cat(global_state_prior_cov_list, dim=1)
        global_state_post_means = torch.cat(global_state_post_mean_list, dim=1)
        global_state_post_covs = torch.cat(global_state_post_cov_list, dim=1)

        ##################################### Decoder ############################################
        ### decode the state to get the observation mean and covariance    要解码成观测还是reward取决于具体任务？
        if self._decode_obs:
            pred_obs_means, pred_obs_covs = self._obsDec(global_state_prior_means, global_state_prior_covs)
            
        if self._decode_reward:
            pred_reward_means, pred_reward_covs = self._rewardDec(global_state_prior_means, global_state_prior_covs)
            
        return pred_obs_means, pred_obs_covs, prior_task_means.detach(), prior_task_covs.detach(), post_task_means.detach(), post_task_covs.detach(), abs_acts.detach()



'''
        for k in range(0,num_episodes): ## first episode is considered too to predict (but usually ignored in evaluation)
            #print("Episode: ", k)
            if k==0:
                state_prior_mean = state_prior_mean_init
                state_prior_cov = state_prior_cov_init
            ### create list of state mean and covariance 
            prior_state_mean_list = []
            prior_state_cov_list = []
            post_state_mean_list = []
            post_state_cov_list = []

            ### get the task post for the current episode (here the assumption is that when observations are missing the task valid flag keeps posterior = prior)
            task_mean = post_task_means[:, k, :]
            task_cov = self._unpack_variances(post_task_covs[:, k, :])   ##将一个tensor分成upper, lower和side

            ### get the obs, action for the current episode
            current_obs_seqs = obs_seqs[:, k*self.H:(k+1)*self.H, :]
            current_act_seqs = action_seqs[:, k*self.H:(k+1)*self.H, :]
            current_obs_valid_seqs = obs_valid_seqs[:, k*self.H:(k+1)*self.H, :]
            current_episode_len = current_obs_seqs.shape[1] # [x] made sure works with episodes < H   ## 是为了最后一个集合吗？

            for t in range(current_episode_len): # [x] made sure works with episodes < H
                ### encode the observation (no time embedding)
                current_obs = current_obs_seqs[:, t, :]

                obs_mean, obs_var = self._obsEnc(current_obs)
                ## expand dims to make it compatible with the update step (which expects a 3D tensor)
                obs_mean = torch.unsqueeze(obs_mean, dim=1)
                obs_var = torch.unsqueeze(obs_var, dim=1)

                ### update the state posterior
                current_obs_valid = current_obs_valid_seqs[:, t, :]
                ## expand dims to make it compatible with the update step (which expects a 3D tensor)
                current_obs_valid = torch.unsqueeze(current_obs_valid, dim=1)
                state_post_mean, state_post_cov = self._obsUpdate(state_prior_mean, state_prior_cov, obs_mean, obs_var, current_obs_valid)

                ### predict the next state mean and covariance using the marginalization layer for worker
                current_act = current_act_seqs[:, t, :]
                mean_list_causal_factors = [state_post_mean, current_act, task_mean]
                cov_list_causal_factors = [state_post_cov, task_cov]
                
                state_next_mean, state_next_cov = self._state_predict(mean_list_causal_factors, cov_list_causal_factors)

                ### update the state prior
                state_prior_mean, state_prior_cov = state_next_mean, state_next_cov  ### this step also makes sure every episode 
                                                                                        ### starts with the prior of the previous episode
                ## 为啥不直接 state_prior_mean, state_prior_cov = self._state_predict(mean_list_causal_factors, cov_list_causal_factors) ？

                ### concat 
                ### append the state mean and covariance to the list
                prior_state_mean_list.append(state_prior_mean)
                prior_state_cov_list.append(torch.cat(state_prior_cov, dim=-1))
                post_state_mean_list.append(state_post_mean)
                post_state_cov_list.append(torch.cat(state_post_cov, dim=-1))
            
            ## detach the state prior mean and covariance to make sure the next episode starts with the prior of the previous episode  将状态先验均值和协方差分离，以确保下一窗口从上一窗口的先验开始
            state_prior_mean = state_prior_mean.detach()
            state_prior_cov = [cov.detach() for cov in state_prior_cov]

            ### stack the list to get the final tensors   将list堆叠张量
            prior_state_means = torch.stack(prior_state_mean_list, dim=1)
            prior_state_covs = torch.stack(prior_state_cov_list, dim=1)
            post_state_means = torch.stack(post_state_mean_list, dim=1)
            post_state_covs = torch.stack(post_state_cov_list, dim=1)

            ### append the state mean and covariance to the list
            global_state_prior_mean_list.append(prior_state_means) 
            global_state_prior_cov_list.append(prior_state_covs)
            global_state_post_mean_list.append(post_state_means)
            global_state_post_cov_list.append(post_state_covs)

        ### concat along the episode dimension
        global_state_prior_means = torch.cat(global_state_prior_mean_list, dim=1)
        global_state_prior_covs = torch.cat(global_state_prior_cov_list, dim=1)
        global_state_post_means = torch.cat(global_state_post_mean_list, dim=1)
        global_state_post_covs = torch.cat(global_state_post_cov_list, dim=1)

        ##################################### Decoder ############################################
        ### decode the state to get the observation mean and covariance    要解码成观测还是reward取决于具体任务？
        if self._decode_obs:
            pred_obs_means, pred_obs_covs = self._obsDec(global_state_prior_means, global_state_prior_covs)
        if self._decode_reward:
            pred_reward_means, pred_reward_covs = self._rewardDec(global_state_prior_means, global_state_prior_covs)

            
        return pred_obs_means, pred_obs_covs, prior_task_means.detach(), prior_task_covs.detach(), post_task_means.detach(), post_task_covs.detach(), abs_acts.detach()
'''