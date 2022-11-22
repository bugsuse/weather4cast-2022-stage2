import os
import h5py
import numpy as np


def tensor_to_submission_file(predictions, predict_params):
    """saves prediction tesnor to submission .h5 file

    Args:
        predictions (numpy array): data cube of predictions
        predict_params (dict): dictionary of parameters for prediction
    """
    path = os.path.join(predict_params["submission_out_dir"],
                        str(predict_params["year_to_predict"]))
    if not os.path.exists(path):
        os.makedirs(path)

    submission_file_name = predict_params['region_to_predict'] + '.pred.h5'
    submission_path = os.path.join(path, submission_file_name)
    h5f = h5py.File(submission_path, 'w')
    h5f.create_dataset('submission', data=predictions.squeeze())
    h5f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("pack_name", type=str, help='the name of prediction to pack')
    parser.add_argument("-years", "--years", type=int, nargs='+', required=False, default=None,
                        help="the years")
    parser.add_argument("-models", "--models", type=str, nargs='+', required=False, default=None,
                        help="the models to ensemble")
    parser.add_argument("-weights", "--weights", type=float, nargs='+', required=False, default=None,
                        help="the weights corresponding to the models to ensemble")
    parser.add_argument("-thresholds", "--thresholds", type=float, required=False, default=None,
                        help="the thresholds")
    options = parser.parse_args()

    if options.years is None:
        years = [2019, 2020]
    else:
        years = options.years

    if options.thresholds is not None:
        thre = options.thresholds
    else:
        thre = 0.4

    regions = {'boxi_0015': thre,
               'boxi_0034': thre,
               'boxi_0076': thre,
               'roxi_0004': thre,
               'roxi_0005': thre,
               'roxi_0006': thre,
               'roxi_0007': thre,
              }

    if options.models is None:
        weights = [1, 1]
        models = ['unet3d.ioudice.ep35', 'unet3d.ioudice.ep47']
    else:
        weights = options.weights
        models = options.models

    for ye in years:
        for reg, thre in regions.items():
            data = np.zeros((60, 32, 252, 252))
            for model, weight in zip(models, weights):
                print(model, weight, ye, reg, thre)
                data += weight*h5py.File(f'{model}/{ye}/{reg}.pred.h5')['submission'][:]

            data /= len(models)

            gt0 = data >= thre
            data[gt0] = 1
            data[~gt0] = 0

            data = data.astype('uint8')

            params = {'submission_out_dir': 'ensemble.core', 'year_to_predict': ye,
                      'region_to_predict': reg}
            tensor_to_submission_file(data, params)
            os.system(f"gzip -9f {params['submission_out_dir']}/{ye}/{reg}.pred.h5")

    os.system(f'cd ensemble.core && zip -r {options.pack_name}.zip 20*')

