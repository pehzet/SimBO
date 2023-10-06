import copy
import sys
import traceback
import logging

logger = logging.getLogger("mrp")

def creategrossReqArray():
    grossReq = []
    for m in g.materials:
        _array = [0] * g.planningHorizon
        grossReq.append({
            'id': str(m.get('id')),
            'values': _array})
    return grossReq


def bom_explode(child_id, quantity, period, parent_id='null'):

    if child_id not in g._already_exploted:
        _safety_stock = [p.get('value', 0) for p in g.parameters if p.get('id') == child_id and p.get('name') == 'safety_stock'][0]

        g._already_exploted.append(child_id)
    else:
        _safety_stock = 0
    _quant = float(quantity) + float(_safety_stock)
    if parent_id == 'null':
        _p_lead_time = 0
    else:
        _p_lead_time = int([m.get("lead_time", 0) for m in g.materials if m.get("id") == parent_id][0]) + int(
            [p.get('value', 0) for p in g.parameters if p.get('id') == parent_id and p.get('name') == 'safety_time'][0])
   
    if int(period) - int(_p_lead_time) < 0:
        logger.warning(f"Period Warning for Material {child_id}. Material Requirement Period is less than 0: {int(period) - int(_p_lead_time)}. Going to set Requirement Period to 1.")
        #print(f"Period Warning for Material {child_id}. Material Requirement Period is less than 0: {int(period) - int(_p_lead_time)}. Going to set Requirement Period to 1.")
        
    _per = max(int(period) - int(_p_lead_time), 1)

    if _per > 0:
        __per = _per - 1
    else:
        __per = _per

    [gr.get('values') for gr in g.grossReq if gr.get('id') == child_id][0][__per] += _quant

    next_level = [x for x in g.bom if str(x.get("parent_id")) == str(child_id)]
    _parent_id = child_id
    for child in next_level:
        bom_explode(child.get("child_id"), _quant *float(child.get("quantity")), _per, _parent_id)
  

def cleangrossReq():
    for gr in g.grossReq:
        if sum(gr.get('values')) == 0:
            g.grossReq.remove(gr)

def calcNetRequirements():
    netReq = copy.deepcopy(g.grossReq)
    for nr in netReq:

        first_event = False

        _safety_stock = float([p.get('value') for p in g.parameters if str(p.get('id')) == str(nr["id"]) and p.get('name') == 'safety_stock'][0])

        for i, e in enumerate(nr.get("values")):

            if e != 0:
                _stock = float([s.get("stock", 0) for s in g.inventory[i] if str(s.get("id")) == nr.get("id")][0])
                if first_event == True : 
                    nr.get("values")[i] = (e - _stock + _safety_stock)
                else:
                    nr.get("values")[i] = (e - _stock)

                g.inventory[i+1].append({"id": nr.get("id"),"stock": _safety_stock})
                first_event = True
            else:
                if first_event == True:
                    g.inventory[i+1].append({"id": nr.get("id"),
                                            "stock": _safety_stock})
                else:
                    _stock = float([s.get("stock", 0) for s in g.inventory[i] if str(
                        s.get("id")) == nr.get("id")][0])
                    g.inventory[i +
                                1].append({"id": nr.get("id"), "stock": _stock})

    return netReq


def releaseOrders():
    releasedOrders = []
    for nr in g.netRequirements:
        _ro = [0] * g.planningHorizon
        _leadTime = int([m.get("lead_time", 0) for m in g.materials if str(m.get("id")) == nr.get('id')][0])
        
        _safety_time = int([e.get('value', 0) for e in g.parameters if str(e.get('id')) == nr.get('id') and e.get('name') == 'safety_time'][0])
        for o in range(len(nr.get('values'))):
            if nr.get('values')[o] > 0:
                if int(o - _leadTime - _safety_time) < 0:
                    logger.warning(f"Period Warning for Material {nr.get('id')}. Material Purchase Period is less than 0: {(o - _leadTime - _safety_time)}. Going to place it at Period 0.")
                    #print(f"Period Warning for Material {nr.get('id')}. Material Purchase Period is less than 0: {(o - _leadTime - _safety_time)}. Going to place it at Period 0.")
                _ro[max(int(o - _leadTime - _safety_time), 0)] = nr.get('values')[o]

        _rod = {'id': nr.get('id'), 'values': _ro}
        releasedOrders.append(_rod)

    return releasedOrders


def getMRPResults():
    _marray = []
    for m in g.materials:
        _mdict = {}
        _m_id = str(m.get('id'))
        _mdict["id"] = _m_id
        _mdict["GrossRequirements"] = [
            x.get('values') for x in g.grossReq if x.get('id') == _m_id][0]
        _mdict["NetRequirements"] = [
            x.get('values') for x in g.netRequirements if x.get('id') == _m_id][0]
        _mdict["ReleasedOrders"] = [
            x.get('values') for x in g.releasedOrders if x.get('id') == _m_id][0]
        _marray.append(_mdict)

    return _marray

# here the Main Logic starts


class g:
    pass

def run_mrp(bom, materials, orders, inventory, parameters, horizon=100):
    logger.debug("Starting MRP run...")
    #print("Starting MRP")
    # first define and get the global variables
    g.planningHorizon = horizon
    g.bom = bom
    g.materials = materials
    g.orders = orders
    g.parameters = parameters
    g._already_exploted = []
    g.inventory = []
    g.inventory.append(inventory)
    for i in range(g.planningHorizon):
        g.inventory.append([])



    g.netRequirements = []
    g.releasedOrders = []
    try:
        g.grossReq = creategrossReqArray()
    except BaseException as e:
        print(traceback.format_exc())
        sys.exit()
    # Now let the Run start:

    # grossReq = creategrossRequirements() # Diese Funktion beinhaltet schon den Demand. Das ist falsch. Dieser wird bei der BOM hinzugefügt, deshalb nur das leere Array erstellen
    try:
        for o in g.orders:
            bom_explode(str(o.get("id")), o.get("quantity"), o.get("period"))
    except BaseException as e:
        logger.error(traceback.format_exc())
        #print(traceback.format_exc())
        sys.exit()
    # cleangrossReq()
    try:
        g.netRequirements = calcNetRequirements()
    except BaseException as e:
        logger.error(traceback.format_exc())
        #print(traceback.format_exc())
        sys.exit()
    try:        
        g.releasedOrders = releaseOrders()
    except BaseException as e:
        logger.error(traceback.format_exc())
        #print(traceback.format_exc())
        sys.exit()
    try:
        mrpResults = getMRPResults()
      
    except BaseException as e:
        logger.error(traceback.format_exc())
        #print(traceback.format_exc())
        sys.exit()


    return mrpResults