from types import SimpleNamespace

from tinker_cookbook.rl import train

from feather_train.recipes.tinker.braintrust import track_training_batches


async def test_unreadable_span_metadata_preserves_successful_training_result(monkeypatch, caplog):
    result = (object(), {}, [SimpleNamespace(trajectory_group=object())])

    async def step(*args, **kwargs):
        return result

    monkeypatch.setattr(train, "do_train_step_streaming_and_get_sampling_client", step)
    with track_training_batches():
        returned = await train.do_train_step_streaming_and_get_sampling_client(None, 3)

    assert returned is result
    assert "Could not read training batch 3 for span tagging" in caplog.text
