import numpy as np
import pytest
import torch
from audit.metrics import cosine, pearson, PredictionMean, panel, paired_panel, postprocess
from audit.flows import (complete_samples, chronological_order, ofi_level, kyle_terms,
                         new_order_gaps, local_mid_reference)
from audit.protocol import jobs
from audit.training import loss_parts, fit_scale, apply_scale, seed_all


def test_metric_shift_and_scale():
    y = np.array([-1.,1.])
    assert pearson(y+10,y) == pytest.approx(1.)
    assert cosine(y+10,y) == pytest.approx(1/np.sqrt(101))
    assert cosine(7*y,y) == pytest.approx(1.)
    assert cosine(np.zeros(2),y) == 0
    with pytest.raises(ValueError):
        cosine([np.nan], [1])


def test_oof_averages_seeds_and_rejects_duplicate_models():
    p = PredictionMean(4)
    p.add('s1',[0,2],[1.,4.]); p.add('s2',[0,2],[3.,8.])
    np.testing.assert_equal(p.mean(),[2.,np.nan,6.,np.nan])
    np.testing.assert_equal(p.count,[2,0,2,0])
    with pytest.raises(ValueError):
        p.add('s2',[0],[9])


def test_clipping_preserves_no_data_zero():
    np.testing.assert_equal(postprocess([0,5],[1,2],[True,False]),[0,2])


def test_month66_is_in_primary_and_paired_delta():
    p=np.array([1.,-1.,-1.,1.]); y=np.array([1.,-1.,1.,-1.]); m=np.array([65,65,66,66])
    r=panel(p,y,m)
    assert r['overall']==pytest.approx(0)
    assert r['without_66_sensitivity']==pytest.approx(1)
    pair=paired_panel(p,y,y,m,bootstrap=20)
    assert pair['delta']==pytest.approx(1)
    assert pair['monthly_delta']['66']==pytest.approx(2)


def test_complete_samples_and_tie_order_across_chunks():
    sid=np.array([0,0,0,1,1,2]); t=np.array([10.,0.,0.,5.,1.,0.]); values=np.arange(6)
    def result(size):
        chunks=((sid[i:i+size],t[i:i+size],values[i:i+size]) for i in range(0,6,size))
        result=[]
        seen=set()
        for s,times,v in complete_samples(chunks):
            assert not seen.intersection(s.tolist())
            seen.update(s.tolist())
            result.extend(v[chronological_order(s,times)].tolist())
        return result
    assert result(1)==result(2)==result(4)==result(100)==list(range(6))
    with pytest.raises(ValueError):
        list(complete_samples([(np.array([1,0]),np.zeros(2))]))


def test_chronological_ofi_hand_calculation_and_empty_book():
    sid=np.zeros(3,int); t=np.array([0.,20.,10.])
    order=chronological_order(sid,t)
    np.testing.assert_equal(order,[1,2,0])
    # At unchanged prices, bid queue +5 then ask queue +3: OFI +5, -3.
    e=ofi_level(np.ones(3),np.array([10.,15.,15.]),np.ones(3)*2,np.array([10.,10.,13.]),np.ones(2,bool))
    np.testing.assert_equal(e,[5.,-3.])
    e=ofi_level(np.array([0.,1.]),np.array([0.,10.]),np.array([2.,2.]),np.array([1.,1.]),np.array([True]))
    assert e[0]==0


def test_kyle_never_counts_cross_sample_or_invalid_price_denominator():
    num,den=kyle_terms(np.array([0,0,1,1]),np.array([1.,2.,0.,3.]),np.array([1.,2.,100.,200.]))
    np.testing.assert_allclose(num,[2*np.log(2),0,0])
    np.testing.assert_equal(den,[4.,0.,0.])


def test_new_gaps_ignore_intervening_cancels():
    sid,gap=new_order_gaps(np.array([0,0,0,1]),np.array([60.,30.,0.,0.]),np.array([True,False,True,True]))
    np.testing.assert_equal(sid,[0]); np.testing.assert_equal(gap,[60.])


def test_mid_reference_is_independent_of_other_samples():
    x=np.array([[1.,0.,2.],[10.,20.,0.]])
    expected=local_mid_reference(x,x>0)
    np.testing.assert_equal(expected,[1.5,15.])
    assert local_mid_reference(x[:1],x[:1]>0)[0]==expected[0]
    assert local_mid_reference(np.zeros((1,3)),np.zeros((1,3),bool))[0]==1


def test_clean_weights_use_original_target_not_noise():
    pred=torch.tensor([[.003],[.003]],requires_grad=True)
    clean=torch.tensor([0.,.003]); noisy=torch.tensor([.002,0.])
    _,mse,_=loss_parts(pred,clean,noisy,'clean')
    _,old,_=loss_parts(pred,clean,noisy,'noisy')
    assert mse.item()==pytest.approx((.001**2+.5*.003**2)/2)
    assert old.item()==pytest.approx((.5*.001**2+.003**2)/2)


def test_scaler_constant_feature_is_finite_and_train_fitted():
    x=np.array([[1.,2.],[1.,4.],[1.,6.]],np.float32)
    med,fac=fit_scale(x)
    result=apply_scale(np.array([[1.,1000.]],np.float32),med,fac)
    assert result[0,0]==0 and np.isfinite(result).all() and result[0,1]<=3


def test_nested_protocol_has_no_outer_labels_in_any_model_selection():
    cfg={'mode':'confirm','arms':['base','x30'],'seeds':[2026,42]}
    jj=jobs(cfg); assert len(jj)==40
    months=np.repeat(np.arange(71),2)
    seen=set()
    for j in jj:
        tr,es,score=j.masks(months)
        assert not np.any((tr|es)&score)
        assert months[es].max()<=j.origin<months[score].min()
        seen.update(months[score].tolist())
    assert 66 in seen
    with pytest.raises(ValueError):
        jobs({**cfg,'folds':[4]})
    with pytest.raises(ValueError):
        jobs({**cfg,'origins':[{'origin':59,'score_start':62,'score_end':67}, {'origin':64,'score_start':67,'score_end':70}]})


def test_cpu_seed_resets_after_unrelated_random_work():
    seed_all(42,torch.device('cpu')); expected=torch.randn(8)
    torch.randn(999)
    seed_all(42,torch.device('cpu'))
    torch.testing.assert_close(expected,torch.randn(8),rtol=0,atol=0)
