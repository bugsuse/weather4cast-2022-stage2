from .baseline_UNET3D import UNet as Unet3d
from .cuboid_transformer import CuboidTransformerModel


def get_model(model_name, model_params):
    from omegaconf import OmegaConf

    if model_name == 'unet3d':
        model = Unet3d(**model_params)
    elif model_name == 'earthformer':
        num_blocks = len(model_params["enc_depth"])
        if isinstance(model_params["self_pattern"], str):
            enc_attn_patterns = [model_params["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_params["self_pattern"])
        if isinstance(model_params["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_params["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_params["cross_self_pattern"])
        if isinstance(model_params["cross_pattern"], str):
            dec_cross_attn_patterns = [model_params["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_params["cross_pattern"])

        model = CuboidTransformerModel(
            input_shape=model_params["input_shape"],
            target_shape=model_params["target_shape"],
            base_units=model_params["base_units"],
            block_units=model_params["block_units"],
            scale_alpha=model_params["scale_alpha"],
            enc_depth=model_params["enc_depth"],
            dec_depth=model_params["dec_depth"],
            enc_use_inter_ffn=model_params["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_params["dec_use_inter_ffn"],
            downsample=model_params["downsample"],
            downsample_type=model_params["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_params["cross_last_n_frames"],
            dec_use_first_self_attn=model_params["dec_use_first_self_attn"],
            num_heads=model_params["num_heads"],
            attn_drop=model_params["attn_drop"],
            proj_drop=model_params["proj_drop"],
            ffn_drop=model_params["ffn_drop"],
            upsample_type=model_params["upsample_type"],
            ffn_activation=model_params["ffn_activation"],
            gated_ffn=model_params["gated_ffn"],
            norm_layer=model_params["norm_layer"],
            # global vectors
            num_global_vectors=model_params["num_global_vectors"],
            use_dec_self_global=model_params["use_dec_self_global"],
            dec_self_update_global=model_params["dec_self_update_global"],
            use_dec_cross_global=model_params["use_dec_cross_global"],
            use_global_vector_ffn=model_params["use_global_vector_ffn"],
            use_global_self_attn=model_params["use_global_self_attn"],
            separate_global_qkv=model_params["separate_global_qkv"],
            global_dim_ratio=model_params["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_params["initial_downsample_type"],
            initial_downsample_activation=model_params["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_params["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_params["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_params["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_params["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_params["padding_type"],
            z_init_method=model_params["z_init_method"],
            checkpoint_level=model_params["checkpoint_level"],
            pos_embed_type=model_params["pos_embed_type"],
            use_relative_pos=model_params["use_relative_pos"],
            self_attn_use_final_proj=model_params["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_params["attn_linear_init_mode"],
            ffn_linear_init_mode=model_params["ffn_linear_init_mode"],
            conv_init_mode=model_params["conv_init_mode"],
            down_up_linear_init_mode=model_params["down_up_linear_init_mode"],
            norm_init_mode=model_params["norm_init_mode"])
    else:
        raise ValueError(f'No support {model_name} model!')

    print(model)

    return model

