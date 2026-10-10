"""Code semantics, gradients and serialization; no downloads or APIs."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from rating_recsys.retrieval.rqvae import ResidualQuantizer, ReviewRQVAE, RQVAEConfig


def test_residual_codes_reconstruct_sum_of_selected_codewords():
    quantizer = ResidualQuantizer(2, 2, 2)
    with torch.no_grad():
        quantizer.books[0].weight.copy_(torch.tensor([[1.,0.],[0.,1.]]))
        quantizer.books[1].weight.copy_(torch.tensor([[0.,0.],[0.,1.]]))
    latent = torch.tensor([[1.2,.9],[-.1,1.2]],requires_grad=True)
    reconstructed,codes,loss = quantizer(latent)
    assert codes.tolist() == [[0,1],[1,0]]
    torch.testing.assert_close(reconstructed,torch.tensor([[1.,1.],[0.,1.]]))
    torch.testing.assert_close(reconstructed,quantizer.decode(codes))
    assert loss.item()>0


def test_straight_through_and_codebook_gradients_are_separate():
    torch.manual_seed(8)
    quantizer = ResidualQuantizer(3,8,4)
    latent = torch.randn(10,4,requires_grad=True)
    reconstructed,_,loss = quantizer(latent)
    reconstructed.sum().backward(retain_graph=True)
    torch.testing.assert_close(latent.grad,torch.ones_like(latent))
    assert all(book.weight.grad is None for book in quantizer.books)
    loss.backward()
    assert all(book.weight.grad is not None and book.weight.grad.abs().sum()>0 for book in quantizer.books)


def test_model_codes_roundtrip_after_save_without_source_vectors(tmp_path):
    torch.manual_seed(42)
    config = RQVAEConfig(input_dimension=8,hidden_dimension=12,latent_dimension=4,codebook_size=8)
    model = ReviewRQVAE(config)
    train = torch.randn(40,8)
    model.set_scaling(train)
    model.eval()
    with torch.no_grad():
        expected,codes,_,_ = model(train)
    destination = tmp_path/'model.pt'
    torch.save(model.state_dict(),destination)
    restored = ReviewRQVAE(config)
    restored.load_state_dict(torch.load(destination,weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored.decode_codes(codes),expected,atol=1e-6,rtol=1e-5)


def test_scaling_rejects_invalid_training_data():
    model = ReviewRQVAE(RQVAEConfig(input_dimension=8))
    with pytest.raises(ValueError):model.set_scaling(torch.full((10,8),float('nan')))
    with pytest.raises(ValueError):model.set_scaling(torch.randn(10,9))
    with pytest.raises(ValueError):model.set_scaling(torch.empty(0,8))
    with pytest.raises(ValueError):model.quantizer.decode(torch.full((1,3),256,dtype=torch.int64))


def test_representation_audit_does_not_share_identical_text_with_fit():
    from types import SimpleNamespace
    from rating_recsys.experiments.rqvae_cli import split_review_groups
    rows = [SimpleNamespace(review_id=index) for index in range(30)]
    texts = {index:f'문장 {index//2}' for index in range(30)}
    train,audit = split_review_groups(rows,texts)
    assert len(train)+len(audit)==30
    assert {texts[int(index)] for index in train}.isdisjoint({texts[int(index)] for index in audit})


def test_profile_pooling_gives_equal_weight_after_review_normalization():
    from types import SimpleNamespace
    from datetime import date
    from rating_recsys.experiments.rqvae_cli import profile_banks
    rows=[SimpleNamespace(review_id=i,user_id=1,restaurant_id=1,event_date=date(2025,1,i)) for i in (1,2)]
    (_,users),(_,items)=profile_banks(rows,rows,np.asarray([[10.,0.],[0.,1.]],dtype=np.float32))
    np.testing.assert_allclose(users,[[2**-.5,2**-.5]],rtol=1e-6)
    np.testing.assert_allclose(items,users)


def test_missing_item_keeps_c5_slot_and_seen_items_stay_excluded():
    from types import SimpleNamespace
    from datetime import date
    from rating_recsys.experiments.rqvae_cli import recommendation_metrics
    history=[SimpleNamespace(review_id=i,user_id=u,restaurant_id=r,event_date=date(2025,1,i))
             for i,u,r in ((1,1,10),(2,2,11),(3,2,12),(4,2,13))]
    review_rows=history[:3]
    vectors=np.asarray([[1.,0.],[0.,1.],[1.,0.]],dtype=np.float32)
    query=SimpleNamespace(query_id='validation:u1',user_id=1,relevance_by_item={12:2})
    _,rankings=recommendation_metrics(history,review_rows,{'raw':vectors},[query],{'validation:u1':(11,13,12)})
    assert rankings['raw_c5_text_order']['validation:u1']==(12,13,11)
    assert all(10 not in row['validation:u1'] for row in rankings.values())


def test_effective_token_input_groups_override_different_long_source_texts():
    from types import SimpleNamespace
    from rating_recsys.experiments.rqvae_cli import split_review_groups
    rows=[SimpleNamespace(review_id=i) for i in range(30)]
    texts={i:f'원문 뒤쪽 차이 {i}' for i in range(30)}
    keys=[str(i//2) for i in range(30)]
    train,audit=split_review_groups(rows,texts,effective_group_keys=keys)
    assert {keys[int(i)] for i in train}.isdisjoint({keys[int(i)] for i in audit})
