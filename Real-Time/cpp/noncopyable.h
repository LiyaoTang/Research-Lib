#ifndef NONCOPYABLE_H
#define NONCOPYABLE_H

// There must be many copy-paste versions of these macros which are same
// things, undefine them to avoid conflict.
#undef DISALLOW_COPY
#undef DISALLOW_ASSIGN
#undef DISALLOW_COPY_AND_ASSIGN
#undef DISALLOW_IMPLICIT_CONSTRUCTORS

// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Put this in the private: declarations for a class to be uncopyable.
#define DISALLOW_COPY(TypeName)                 \
    TypeName(const TypeName&)

// Put this in the private: declarations for a class to be unassignable.
#define DISALLOW_ASSIGN(TypeName)               \
    void operator=(const TypeName&)

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName)      \
    TypeName(const TypeName&);                  \
    void operator=(const TypeName&)

// A macro to disallow all the implicit constructors, namely the
// default constructor, copy constructor and operator= functions.
//
// This should be used in the private: declarations for a class
// that wants to prevent anyone from instantiating it. This is
// especially useful for classes containing only static methods.
#define DISALLOW_IMPLICIT_CONSTRUCTORS(TypeName)        \
    TypeName();                                         \
    DISALLOW_COPY_AND_ASSIGN(TypeName)

#endif // NONCOPYABLE_H
