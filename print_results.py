
def get_test_scores(filename):
    with open('/scratch/cluster/tansey/tmp/' + filename, 'rb') as f:
        scores = []
        for line in f:
            if line.startswith('Test score:'):
                scores.append(float(line[len('Test score: '):-1]))
    return scores

if __name__ == '__main__':
    sizes = [256, 512, 1024, 2048, 4096]
    json_files = [256, 512, 1024, 2048, 'big']
    lams = [0.1, 0.01, 0.001]
    lamnames = ['1', '01', '001']

    
    for size, json in zip(sizes, json_files):
        multinomial_filename = '../tmp/out_--data_dir_data_--prob_model_type_softmax_--test_freq_3000_--silence_threshold_0.01_--sample_size_16000_--wavenet_params_wavenet_params_{json}.json_--logdir_-scratch-cluster-tansey-tensorflow-wavenet-results-{size}_softmax.out'
        multinomial_scores = get_test_scores(multinomial_filename.format(size=size, json=json))

        unsmoothed_filename = 'out_--data_dir_data_--prob_model_type_sdp_--test_freq_3000_--silence_threshold_0.01_--sample_size_16000_--wavenet_params_wavenet_params_{json}.json_--logdir_-scratch-cluster-tansey-tensorflow-wavenet-results-{size}_sdp0_--sdp_lam_0.out'
        unsmoothed_scores = get_test_scores(unsmoothed_filename.format(size=size, json=json))

        lam_5_scores = []
        for lam,lamname in zip(lams, lamnames):
            lam_filename = 'out_--data_dir_data_--prob_model_type_sdp_--test_freq_3000_--silence_threshold_0.01_--sample_size_16000_--wavenet_params_wavenet_params_{json}.json_--logdir_-scratch-cluster-tansey-tensorflow-wavenet-results-{size}_sdp{lamname}_--sdp_lam_{lam}.out'
            lam_5_scores.append(get_test_scores(lam_filename.format(size=size, json=json, lam=lam, lamname=lamname)))

        lam_15_scores = []
        for lam,lamname in zip(lams, lamnames):
            if size > 1024 or (size == 1024 and lamname == '001'):
                lam_filename = 'out_--data_dir_data_--prob_model_type_sdp_--test_freq_3000_--silence_threshold_0.01_--sample_size_16000_--wavenet_params_wavenet_params_{json}.json_--logdir_results-{size}_sdp{lamname}_15_--sdp_lam_{lam}_--sdp_radius_15.out'
            else:
                lam_filename = 'out_--data_dir_data_--prob_model_type_sdp_--test_freq_3000_--silence_threshold_0.01_--sample_size_16000_--wavenet_params_wavenet_params_{json}.json_--logdir_-scratch-cluster-tansey-tensorflow-wavenet-results-{size}_sdp{lamname}_15_--sdp_lam_{lam}_--sdp_radius_15.out'
            lam_15_scores.append(get_test_scores(lam_filename.format(size=size, json=json, lam=lam, lamname=lamname)))

        print '********** {} **********'.format(size)
        for i,softmax_score in enumerate(multinomial_scores):
            print '{} Epochs'.format((i+1)*3000)
            print 'Softmax: {:.4f}'.format(softmax_score)
            if len(unsmoothed_scores) > i:
                print 'Unsmoothed: {:.4f}'.format(unsmoothed_scores[i])
            for lam,scores in zip(lams, lam_5_scores):
                if len(scores) > i:
                    print 'Radius 5, Lambda {}: {:.4f}'.format(lam, scores[i])
            for lam,scores in zip(lams, lam_15_scores):
                if len(scores) > i:
                    print 'Radius 15, Lambda {}: {:.4f}'.format(lam, scores[i])
            print ''
        print ''
