import os
import logging


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
    )

    ename = "data/all_features_ns_full"
    cfl = [
        "content_he_mean_answered_correctly",
        "mul_user_content_ema_answ",
        "user_ema_answered_correctly",  #
        "content_mean_answered_correctly",  #
        "abs_chng_timestamp_1",
        "attempt_no_count",
        "abs_chng_timestamp_2",
        "user_sum_answered_correctly",
        "part_bundle_id",
        "strike",
        "prior_question_elapsed_time",
        "attempt_no_mean",
        "user_he_mean_prior_question_elapsed_time",
        "rel_user_he_mean_answ",
        "user_he_mean_answered_correctly",  #
        "user_mean_answered_correctly",  #
        "content_he_part_count_answered_correctly",
        "content_sum_answered_correctly",
        "abs_chng_timestamp_3",
        "lag_expl_and_part_bool",
        "rel_user_content_he_mean_answ",
        "user_he_ucount_part",
        "trend_user_he_mean_answ",
        "user_he_ema_answered_correctly",  #
        "strike_bundle",
    ]
    # cfl = []
    if len(cfl) > 0:
        cfl = f' -cfl {",".join(cfl)}'
    else:
        cfl = ""

    """os.system(f"python 0_0_create_vitrine.py -d -in_name {'data/'} -out_name {ename}")

    logging.info("")
    os.system(
        f"python 0_3_get_samples.py -d -in_name {ename} -out_name {ename} \
        -count_train None -count_val None -val_ratio 0.1"
    )

    logging.info("")
    os.system(
        f"python 1_1_generate_user_features.py -d -in_name {ename} -out_name {ename}{cfl}"
    )"""

    logging.info("")
    os.system(
        f"python 1_2_generate_content_features.py -d -in_name {ename} -out_name {ename}{cfl}"
    )

    logging.info("")
    os.system(
        f"python 2_generate_combinations.py -d -in_name {ename} -out_name {ename}{cfl}"
    )
